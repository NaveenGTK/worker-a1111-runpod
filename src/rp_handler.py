import time

import runpod
import requests
import os
from requests.adapters import HTTPAdapter, Retry

LOCAL_URL = "http://127.0.0.1:3000/sdapi/v1"

automatic_session = requests.Session()
retries = Retry(total=10, backoff_factor=0.1, status_forcelist=[502, 503, 504])
automatic_session.mount('http://', HTTPAdapter(max_retries=retries))


# ---------------------------------------------------------------------------- #
#                              Automatic Functions                             #
# ---------------------------------------------------------------------------- #
def wait_for_service(url):
    '''
    Check if the service is ready to receive requests.
    '''
    while True:
        try:
            requests.get(url, timeout=120)
            return
        except requests.exceptions.RequestException:
            print("Service not ready yet. Retrying...")
        except Exception as err:
            print("Error: ", err)

        time.sleep(0.2)


def download_lora(lora_download_link, lora_name, destination_folder):
    """
    Download the LoRA file from the provided link and save it to the destination folder with the specified name.
    """
    try:
        destination_path = os.path.join(destination_folder, lora_name)

        # Ensure the destination folder exists
        os.makedirs(destination_folder, exist_ok=True)

        # Download the file
        response = requests.get(lora_download_link, stream=True)
        response.raise_for_status()  # Raise an error for HTTP issues

        # Save the file
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"LoRA file downloaded and saved to: {destination_path}")
        return destination_path

    except Exception as e:
        print(f"Error downloading LoRA: {e}")
        raise

def run_inference(inference_request):
    """
    Run the inference session.
    """
    try:
        # Run the inference session
        response = automatic_session.post(url=f'{LOCAL_URL}/txt2img',
                                          json=inference_request, timeout=600)

        print("Inference completed.")
        return response.json()

    except Exception as e:
        print(f"Error during inference: {e}")
        raise


# ---------------------------------------------------------------------------- #
#                                RunPod Handler                                #
# ---------------------------------------------------------------------------- #
def handler(event):
    '''
    This is the handler function that will be called by the serverless.
    '''

    input_data = event["input"]
    lora_link = input_data["lora_link"]
    lora_name = input_data["lora_name"]

    # Define the destination folder for LoRA files
    destination_folder = '/stable-diffusion-webui/models/Lora'

    # Download the LoRA file
    download_lora(lora_link, lora_name, destination_folder)

    json = run_inference(event["input"])

    # return the output that you want to be returned like pre-signed URLs to output artifacts
    return json


if __name__ == "__main__":
    wait_for_service(url=f'{LOCAL_URL}/txt2img')

    print("WebUI API Service is ready. Starting RunPod...")

    runpod.serverless.start({"handler": handler})
