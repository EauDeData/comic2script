import os
import pymongo
import numpy as np
import pickle
from bson.binary import Binary
import pandas as pd
import os
import gridfs
import torch.nn as nn

# TODO: This is Re-Used code; adapt for this case and erase things we dont need.

db_name = "comics"

# https://open.spotify.com/track/1u8c2t2Cy7UBoG4ArRcF5g?si=2870274994804085


def list_comics(base):

    comics = {}

    for (dir_path, dir_names, file_names) in os.walk(base):

        for file in file_names:

            splitted = dir_path.split('/')
            comic_volume, comic_edition = splitted[-3], splitted[-2]
            if not comic_volume in comics: comics[comic_volume] = {}
            if not comic_edition in comics[comic_volume]: comics[comic_volume][comic_edition] = []
            comics[comic_volume][comic_edition] += [file]

    
    return comics


def get_mongo_instace(endpoint = 'mongodb://localhost:27017/'):
    return pymongo.MongoClient(endpoint) # Given the mongo service endpoint returns the client instance

def get_or_create_db(name, client):
    return client[name]

def list_db(client):
    return client.list_database_names()

def drop_db(name, client):
    return client.drop_database(name)

def get_or_create_collection(name, db):
    return db[name]

def get_collection(db_name, collection_name, client):
    return get_or_create_collection(collection_name, get_or_create_db(db_name, client))

def upload_serialized_media(media, client, db_name, collection_name, meta, identifier):

    binary_file = Binary(pickle.dumps(media, protocol=2))
    collection = get_collection(db_name, collection_name, client)
    fs = gridfs.GridFS(get_or_create_db(db_name, client))
    id_ = fs.put(binary_file)
    data = {'data': id_, identifier: meta[identifier], 'meta': meta}
    collection.insert_one(data)
    return fs

def upload_serialized_and_get_id(media, client, db_name, collection_name):

    binary_file = Binary(pickle.dumps(media, protocol=2))
    collection = get_collection(db_name, collection_name, client)
    fs = gridfs.GridFS(get_or_create_db(db_name, client))
    id_ = fs.put(binary_file)

    return id_

def download_serialized_from_id(id_, client, db_name, collection_name = None):
    return pickle.loads(gridfs.GridFS(get_or_create_db(db_name, client)).get(id_).read())


def upload_image_db(image, client, db_name, collection_name, meta):
    

    ######### UPLOAD IMAGE ###########
    # Uploads image to db            #
    # With meta fields               #  
    # Some important meta fields     #
    #  * imName: Mandatory           #
    #  * Collection: Mandatory       #
    #  * AnnoyIndex: Mandatory       #
    #  * gtLabel: Optional           #
    #  * PredictedLabel: Optional    #
    #  * Source: Optional            #
    ##################################

    return upload_serialized_media(image, client, db_name, collection_name, meta, 'img_name')

def upload_model(model, client, db_name, collection_name, meta = {}):
    return upload_serialized_media(model, client, db_name, collection_name, meta, 'model_name')

def retrieve_data_document_by_query(query, db_name, collection_name, client, unpickle = True):
    data = list(get_collection(db_name, collection_name, client).find(query))
    if unpickle:
        fs = gridfs.GridFS(get_or_create_db(db_name, client))
        for sample in data: sample['data'] = pickle.loads(fs.get(sample['data']).read())
        return data
    return data

def retrieve_image_by_query(query, db_name, client, unpickle = True):
    return retrieve_data_document_by_query(query, db_name, 'images', client, unpickle=unpickle)

def retrieve_images_by_collection(query, db_name, client):
    return retrieve_image_by_query({"meta.collection": query}, db_name, client)

def update_document(db_name, collection_name, client, meta, id_, identifier = 'img_name'):
    collection = get_collection(db_name, collection_name, client)
    key = f'meta.{identifier}'
    return collection.update_many({key: id_}, {"$set": meta})


def upload_tree_document_lut(data, db_name, client):

    #######################################################################
    # Should introduce a lut so img_identifier ---> AnnoyIndex, AnnoyTree #
    # Data: model_name, collection_name, img_identifier, annoy_index      #
    #######################################################################

    collection = get_collection(db_name, 'annoy', client) # TODO: Preconditions for data having the indicated keys

    return collection.insert_one(data)

def add_collection_csv(db_name, collection_name, client, dataframe, identifier = 'img_name'):
    
    records = dataframe.to_dict('records')
    collection = get_collection(db_name, collection_name, client)
    key = f'meta.{identifier}'
    for sample in records:
        query_sample = "{}".format(sample[identifier])
        meta = {f"meta.{x}": sample[x] for x in sample}
        collection.find_one_and_update({key: query_sample}, {"$set": meta} )  # Many or One?
        
    return None

def get_image_collections(db_name, client):

    images_collection = get_collection(db_name, 'images', client)
    return images_collection.find().distinct('meta.collection')

if __name__ == '__main__':

    print(list_comics('/home/adri/Desktop/cvc/data/comics/comicbookplus_data/'))