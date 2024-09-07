from transformers import AutoModel, AutoTokenizer
import gradio as gr
import torch
from PIL import Image

model = AutoModel.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()


def predict(images, text_input):
    output = ""
    i=1
    question = '''Give a detailed, step-by-step guide on how to test each functionality. Each test case should include:
                    Description: What the test case is about.
                    Pre-conditions: What needs to be set up or ensured before testing.
                    Testing Steps: Clear, step-by-step instructions on how to perform the test.
                    Expected Result: What should happen if the feature works correctly.'''
    if text_input:
      context_question = f"Context : {text_input} /n {question}" 
    else:
      context_question = question
    image1 = Image.open('./prompt_imgs/swiggy_prompt.jpg').convert('RGB')
    answer1 = """
    Test Case 1: Location Setting

    Description
    This test verifies that users can set and change their delivery location.

    Pre-conditions
    The app is installed and launched on a mobile device
    The user has granted location permissions to the app

    Testing Steps
    Open the app
    Tap on the location dropdown at the top of the screen (currently showing "Locating...")
    Enter a new address: "Kolupukur Kumar Para, Surerpukur, Chandannagar"
    Select the entered address from the suggestions

    Expected Result
    The location at the top of the screen should update to display the new address
    The app's content (offers, restaurants) should refresh to reflect the new location


    Test Case 2: Search Functionality

    Description
    This test ensures that the search feature works correctly for finding specific items or cuisine.

    Pre-conditions
    The app is open and on the main screen

    Testing Steps
    Tap on the search bar that says "Search for 'Onam Special'"
    Type "Pizza" into the search bar
    Tap the search icon or press enter on the keyboard

    Expected Result
    Search results should appear, showing restaurants or menu items related to pizza
    The previous content should be replaced with search results


    Test Case 3: Promotional Banner Interaction

    Description
    This test verifies that the promotional banner is interactive and leads to the correct offer page.

    Pre-conditions
    The app is open and on the main screen

    Testing Steps
    Locate the promotional banner (currently showing "Up to 60% OFF")
    Tap on the "ORDER NOW" button within the banner

    Expected Result
    The app should navigate to a new page showing details of the late-night offer
    The new page should display menu items eligible for the up to 60% discount


    Test Case 4: Quick Access Features

    Description
    This test checks the functionality of the quick access buttons below the promotional banner.

    Pre-conditions
    The app is open and on the main screen

    Testing Steps
    Locate the "OFFER ZONE" button
    Tap on the "OFFER ZONE" button
    Return to the main screen
    Locate and tap on the "FOOD ON TRAIN" button

    Expected Result
    Tapping "OFFER ZONE" should open a page with current offers and discounts
    Tapping "FOOD ON TRAIN" should open a page for ordering food for train journeys


    Test Case 5: Restaurant Selection

    Description
    This test ensures users can view restaurant details and select a restaurant for ordering.

    Pre-conditions
    The app is open and on the main screen

    Testing Steps
    Scroll down to the restaurant listings
    Find "Domino's Pizza" in the list
    Tap on the Domino's Pizza card

    Expected Result
    A new page should open showing Domino's Pizza menu and details
    The page should display the offer (₹150 OFF ABOVE ₹299) and delivery time (25-30 mins)


    Test Case 6: Navigation Bar Functionality

    Description
    This test verifies that the bottom navigation bar works correctly for switching between main app sections.

    Pre-conditions
    The app is open and on the main screen

    Testing Steps
    Locate the bottom navigation bar
    Tap on the "Dineout" icon
    Tap on the "Reorder" icon
    Tap on the "Wine Stores" icon
    Tap on the "Genie" icon
    Tap on the "Food" icon to return to the main screen

    Expected Result
    Each tap should navigate to the corresponding section of the app
    The "Food" icon should return the user to the main food ordering screen
    The selected icon in the navigation bar should be highlighted to indicate the current section
    """

    image2 = Image.open('./prompt_imgs/chrome_prompt.jpg').convert('RGB')
    answer2 = """
    Test Case 1: Search Functionality

    Description
    This test verifies that users can perform a search using the search bar.

    Pre-conditions
    The browser app is installed and launched on a mobile device
    The device has an active internet connection

    Testing Steps
    Tap on the search bar that says "Search or type URL"
    Type a search query, e.g., "Kotaemon RAG tool"
    Tap the search icon or press enter on the keyboard

    Expected Result
    The browser should display search results related to the entered query
    The results page should load within a reasonable time frame


    Test Case 2: Voice Search

    Description
    This test ensures that the voice search feature works correctly.

    Pre-conditions
    The browser app is open on the main screen
    The device has a working microphone
    The app has microphone permissions

    Testing Steps
    Tap on the microphone icon in the search bar
    When prompted, speak a search query clearly, e.g., "Open source RAG tools"
    Wait for the voice input to be processed

    Expected Result
    The spoken words should appear in the search bar
    The browser should perform a search using the voice input
    Search results related to the spoken query should be displayed


    Test Case 3: Shortcut Icons Functionality

    Description
    This test verifies that the shortcut icons below the search bar work correctly.

    Pre-conditions
    The browser app is open on the main screen

    Testing Steps
    Locate the shortcut icons below the search bar
    Tap on the "GitHub" icon
    Wait for the page to load
    Use the back button to return to the main screen
    Tap on the "Upwork" icon

    Expected Result
    Tapping the "GitHub" icon should open the GitHub website
    Tapping the "Upwork" icon should open the Upwork website
    Each website should load correctly within the browser


    Test Case 4: Discover Feed Interaction

    Description
    This test checks the functionality of the Discover feed articles.

    Pre-conditions
    The browser app is open on the main screen
    The Discover feed is visible

    Testing Steps
    Scroll down to the Discover feed if not immediately visible
    Locate the article titled "Kotaemon: An Open-Source RAG-based Tool for Chatting with Your Documents"
    Tap on the article

    Expected Result
    The full article or its source page should open
    The content should be readable and properly formatted
    Any images or media in the article should load correctly


    Test Case 5: Share Functionality

    Description
    This test ensures users can share content from the Discover feed.

    Pre-conditions
    The browser app is open on the main screen
    The Discover feed is visible

    Testing Steps
    Locate the share icon (curved arrow) next to the "Kotaemon" article
    Tap on the share icon
    Select a sharing method from the options presented (e.g., copy link, share to another app)

    Expected Result
    A sharing menu or options should appear
    The selected sharing method should execute correctly (e.g., link copied, sharing interface of another app opens)


    Test Case 6: Navigation Controls

    Description
    This test verifies that the bottom navigation controls work correctly.

    Pre-conditions
    The browser app is open and a web page or article is loaded

    Testing Steps
    Locate the bottom navigation bar
    Tap the home icon (house shape) on the left
    Tap the circular button in the center
    Tap the back button on the left of the navigation bar

    Expected Result
    Tapping the home icon should return to the browser's main page
    Tapping the circular button should open the tab switcher or overview
    Tapping the back button should navigate to the previous page in the browsing history
    """
    for image in images:
        image = Image.open(image).convert('RGB')
        msgs = [
                {'role': 'user', 'content': [image1, question]}, {'role': 'assistant', 'content': [answer1]},
                {'role': 'user', 'content': [image2, question]}, {'role': 'assistant', 'content': [answer2]},
                {'role': 'user', 'content': [image, context_question]}
                ]

        res = model.chat(
            image=None,
            msgs=msgs,
            tokenizer=tokenizer
        )
        output+=f'Image {i}: \n \n {res} \n \n \n'
        i+=1
    return output

with gr.Blocks() as demo:
    gr.Markdown("# Image to Test Case")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            image_input = gr.File(file_count="multiple", label="Upload Images")
            gallery = gr.Gallery(
                label="Uploaded Images",
                show_label=True,
                elem_id="gallery",
                columns=[2],
                rows=[2],
                object_fit="contain",
                height="auto"
            )
            text_input = gr.Textbox(label="Optional Context")
            predict_button = gr.Button("Describe Testing Instructions")

        with gr.Column(scale=1):
            gr.Markdown("## Output")
            output = gr.Textbox(label="Predicted Testing Instructions")

    def update_gallery(files):
        return [file.name for file in files] if files else []

    image_input.change(fn=update_gallery, inputs=[image_input], outputs=[gallery])
    predict_button.click(fn=predict, inputs=[image_input, text_input], outputs=[output])

demo.launch(debug=True, share=True)