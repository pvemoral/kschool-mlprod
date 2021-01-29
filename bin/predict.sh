MODEL_NAME="class_20210129"
INPUT_DATA_FILE="data/instance.json"
VERSION_NAME="v0_1"
REGION="europe-west1"

gcloud ai-platform predict --model $MODEL_NAME \
--version $VERSION_NAME \
--json-instances $INPUT_DATA_FILE \
--region $REGION
