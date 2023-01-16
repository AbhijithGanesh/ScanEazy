from api.logger import logger
import jsonschema
import json
from jsonschema import validate
from api.constants import SCHEMA_DEFAULTS_PATH


def load_json(path, **rest) -> dict:
    with open(path, "r") as fileObj:
        data = json.load(fileObj, **rest)
    return data


def validate_json(json_data, template_path):
    execute_api_schema = load_json(SCHEMA_DEFAULTS_PATH)
    logger.info("Validating JSON Schema")

    try:
        validate(instance=json_data, schema=execute_api_schema)

    except jsonschema.exceptions.ValidationError:
        logger.error(f"Provided Organizer JSON is Invalid: {template_path}")
        return Exception("Invalid Schema")

    message = "JSON Validation Succesful 🟢"
    return True, message
