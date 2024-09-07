# Image_to_Test

A tool that uses a multimodal LLM to describe testing instructions for the features of any digital product, based on screenshots.

## Usage

Clone the repo

```bash
git clone https://github.com/Saptarshi-Bandopadhyay/image_to_test.git
cd image_to_test
```

Install dependencies

```python
pip install -r requirements.txt
```

```python
pip install flash_attn
```

Run gradio app

```python
python3 app.py
```

## Details

The tool uses the MiniCPM-V-2_6-int4 model for its multimodal capabilities. The int4 quantized model has been selected for better inference time.

Multishot prompting is used to generate more accurate and reliable test cases for all functionalities present in the given image of the app.

Gradio is used for the frontend interface.

## Prompting strategy

Two prompts have been used for the model. The screenshots are taken from the Swiggy and Google Chrome mobile apps (located in the prompt_imgs directory).

Along with the images, a text prompt is provided, which includes the following sections for each image:

- Description: What the test case is about.
- Pre-conditions: What needs to be set up or ensured before testing.
- Testing Steps: Clear, step-by-step instructions on how to perform the test.
- Expected Result: What should happen if the feature works correctly.

## Example Inference

Inference is performed on screenshots from the RedBus mobile app (located in the example_imgs directory), providing accurate and reliable test cases.
