# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Please specify your Google Cloud Storage bucket here
# GCS_BUCKET="gs://my-bucket/"
BUCKET=$GCS_BUCKET
TRY_NAME=$NAME

TRAINER_PACKAGE_PATH="./src"
MAIN_TRAINER_MODULE="src.word2vec"

now=$(date +"%Y%m%d_%H%M%S")
JOB_NAME="word_"$TRY_NAME"_$now"

JOB_DIR=$BUCKET$JOB_NAME

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --package-path $TRAINER_PACKAGE_PATH \
    --module-name $MAIN_TRAINER_MODULE \
    --python-version 3.5 \
    --region us-west1 \
    --runtime-version 1.10 \
    --config "config.yaml" \
    -- \
    --output-dir "gs://slideshow_storage/b_out" \
    --output-name "b_data" \
    --input-file "gs://slideshow_storage/b_lovely_landscapes.txt" \
    --log-dir "gs://slideshow_storage/b_log" \
    --batch-size 10 \
    --n-epoch 10000 \
    --embedding-dim 5
