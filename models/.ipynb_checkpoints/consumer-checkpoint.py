import os
import cv2
import time
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from threading import Thread
from queue import Queue
from functools import lru_cache
import redis

import configs
from yolov8 import Yolov8SegmentationTorchscript, Yolov8SegmentationONNX, Yolov8Generator
from yolov8.utils import map_coords, export_detections_to_image, export_detections_to_table

from sqlalchemy import create_engine, Column, Float, String, Integer, Table, MetaData
from sqlalchemy import insert, select
from sqlalchemy import or_, and_
# from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from app_annotation import Annotation, Cache

batch_size = 4
# registry = "HDYolo"
# registry = "yolov8-lung-nuclei"
# configs = Yolov8SegmentationConfig(registry)
# service = Yolov8SegmentationTorchscript(configs, device='cuda:0')
service = Yolov8SegmentationONNX(configs, device='cpu')
# tmp_image_dir = 'tmp_files/imgs'
# if not os.path.exists(tmp_image_dir):
#     os.makedirs(tmp_image_dir)


def test_run(self, image="http://images.cocodataset.org/val2017/000000039769.jpg"):
    print(f"Testing service for {image}")
    st = time.time()
    image = Image.open(requests.get(image, stream=True).raw)
    r = service([np.array(image)])

    print(f"{generated_text} ({time.time()-st}s)")

## Connect to redis queue
# pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)
# async def get_redis():
#     return redis.Redis(connection_pool=pool)

client = redis.Redis(host='localhost', port=6379)
print(f"Ping successful: {client.ping()}")

# SQLAlchemy setup
DATABASE_DIR = "./databases/"

# create a result queue
result_queue = Queue()

# @lru_cache
# def get_db_engine(image_id):
#     db_name = f"{image_id}.db"
#     database_url = f"sqlite:///{os.path.join(DATABASE_DIR, db_name)}"
#     engine = create_engine(database_url, echo=False, future=True)

#     return engine

@lru_cache
def get_sessionmaker(image_id):
    db_name = f'{image_id}.db'
    database_url = f"sqlite:///{os.path.join(DATABASE_DIR, db_name)}"
    engine = create_engine(database_url, echo=False, future=True)
    session = sessionmaker(engine, expire_on_commit=False)

    return session

# def get_session(image_id):
#     session = get_sessionmaker(image_id)()
#     try:
#         yield session
#     except Exception:
#         session.rollback()
#         raise
#     finally:
#         session.close()

def convert_result_to_df(r):
    image_id, registry, tile_id, patch_info, output = r
    output = map_coords(output, patch_info)

    ## save tables
    st = time.time()
    df = export_detections_to_table(
        output, labels_text=configs.labels_text,
        save_masks=True,
    )
    df['xc'] = (df['x0'] + df['x1']) / 2
    df['yc'] = (df['y0'] + df['y1']) / 2
    # df['box_area'] = (df['x1'] - df['x0']) * (df['y1'] - df['y0'])
    df['description'] = 'score=' + df['score'].astype(str)
    df = df.drop(columns=['score'])
    df['annotator'] = registry
    print(f"Export table: {time.time() - st}s.")

    return image_id, registry, tile_id, df


# Function to export results to SQLite database,
# Don't use two threading has bug and super slow currently
# def export_to_db():
#     while True:
#         if not result_queue.empty():
#             image_id, df = result_queue.get()
#             lock = client.lock(f"db_lock:{image_id}")
#             acquired = lock.acquire(blocking=True, blocking_timeout=3)
#             print(f"db_lock:{image_id} ({lock}) acquired={acquired}.")

#             if acquired:
#                 try:
#                     print(f"db_lock:{image_id} ({lock}) got locked.")
#                     with lock:
#                         engine = get_db_engine(image_id)
#                         df.to_sql('annotation', con=engine, if_exists='append', index=False)
#                     print({"message": "Write successful!"})
#                 except:
#                     # for whatever reason failed, we put the results back
#                     result_queue.put((image_id, df))
#                     print({"message": "Failed to write and put back!"})
#                 finally:
#                     # Release the lock
#                     lock.release()
#                     print(f"db_lock:{image_id} ({lock}) got released.")
#             else:
#                 # put results back
#                 result_queue.put((image_id, df))
#                 print({"message": f"Failed to acquire `db_lock:{image_id}`. Processed by another process/thread."})

def entry_exists_in_cache(query, session):
    stmt = select(Cache).where(and_(Cache.registry == query['registry'], Cache.tile_id == query['tile_id']))
    result = session.execute(stmt)

    return result.one_or_none()

def export_to_db(entry):
    image_id, registry, tile_id, df = entry

    lock = client.lock(f"db_lock:{image_id}")
    acquired = lock.acquire(blocking=True, blocking_timeout=3)
    print(f"db_lock:{image_id} ({lock}) acquired={acquired}.")

    if acquired:
        session = get_sessionmaker(image_id)()
        query = {'registry': registry, 'tile_id': tile_id}
        try:
            print(f"db_lock:{image_id} ({lock}) got locked.")
            # engine = get_db_engine(image_id)
            # df.to_sql('annotation', con=engine, if_exists='append', index=False)  # chunksize = 1000
            if not entry_exists_in_cache(query, session):
                session.execute(insert(Annotation), df.to_dict(orient='records'))
                session.execute(insert(Cache), query)
                print(f"Insert into db:{image_id}: {len(df)} entries.")
            else:
                print(f"db_cache:{image_id} already analyzed query: {query}.")
            session.commit()
            return {"message": "Write successfully!", "status": 1}
        except:
            session.rollback()
            return {"message": "Failed to write!", "status": 0}
        finally:
            session.close()
            lock.release()
            print(f"db_lock:{image_id} ({lock}) got released.")
    else:
        return {"message": f"Failed to acquire `db_lock:{image_id}`.", "status": -1}

# def bytes2numpy(image_bytes):
#     # cv2 slightly faster
#     cv2_decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
#     cv2_decoded = cv2.cvtColor(cv2_decoded, cv2.COLOR_BGR2RGB)
#     # pil_decoded = np.array(Image.open(io.BytesIO(image_bytes)))

#     return cv2_decoded


def run(max_halt=None, max_latency=0.5, max_write_attempts=5):
    global_running = True
    max_halt = max_halt or float('inf')
    batch_inputs = []
    st = time.time()

    while time.time() - st < max_halt and global_running:
        if client.exists(registry):
            # try:
            serialized_entry = client.rpop(registry)
            entry = pickle.loads(serialized_entry)
            entry = (entry['image_id'], entry['registry'], entry['tile_id'], 
                     entry['info']['roi_slide'], entry['img'],)
            # entry = json.loads(serialized_entry)
            # entry = (entry['image_id'], entry['registry'], entry['info']['roi_slide'], bytes2numpy(entry['img'])
            batch_inputs.append(entry)
            print(f"Retrieve entry from queue (size={client.llen(registry)}): {len(batch_inputs)}")
            # except:
            #     continue
        else:
            time.sleep(0.1)

        if len(batch_inputs) == batch_size or (len(batch_inputs) > 0 and time.time() - st > max_latency):
            image_ids, registries, tile_ids, patch_infos, images = zip(*batch_inputs)  # pil_images
            pst = time.time()
            outputs = service(images, preprocess=True)
            print(f"Inference batch (size={len(images)}): {time.time() - pst}s.")

            pst = time.time()
            for r in zip(image_ids, registries, tile_ids, patch_infos, outputs):
                # result_queue.put(convert_result_to_df(r))
                r = convert_result_to_df(r)
                status, attempts = -1, 0
                while status <= 0 and attempts < max_write_attempts:
                    response = export_to_db(r)
                    status = response["status"]
                    if status <= 0:
                        time.sleep(0.1)
                        attempts += 1
                print(response)
                # engine = get_db_engine(image_id)
                # df.to_sql('annotation', con=engine, if_exists='append', index=False)
                # print({"message": "Write successful!"})
            print(f"Write to db batch (size={len(images)}): {time.time() - pst}s.")

            batch_inputs = []
            st = time.time()


if __name__ == "__main__":
    run()
#     redis_thread = Thread(target=run)
#     db_thread = Thread(target=export_to_db)

#     redis_thread.start()
#     db_thread.start()
