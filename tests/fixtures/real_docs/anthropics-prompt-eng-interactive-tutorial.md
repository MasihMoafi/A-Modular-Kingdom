Directory structure:
└── anthropics-prompt-eng-interactive-tutorial/
    ├── README.md
    ├── AmazonBedrock/
    │   ├── README.md
    │   ├── requirements.txt
    │   ├── anthropic/
    │   │   ├── 00_Tutorial_How-To.ipynb
    │   │   ├── 01_Basic_Prompt_Structure.ipynb
    │   │   ├── 02_Being_Clear_and_Direct.ipynb
    │   │   ├── 03_Assigning_Roles_Role_Prompting.ipynb
    │   │   ├── 04_Separating_Data_and_Instructions.ipynb
    │   │   ├── 05_Formatting_Output_and_Speaking_for_Claude.ipynb
    │   │   ├── 06_Precognition_Thinking_Step_by_Step.ipynb
    │   │   ├── 07_Using_Examples _Few-Shot_Prompting.ipynb
    │   │   ├── 10_1_Appendix_Chaining_Prompts.ipynb
    │   │   ├── 10_2_Appendix_Tool_Use.ipynb
    │   │   ├── 10_3_Appendix_Empirical_Performance_Evaluations.ipynb
    │   │   └── 10_4_Appendix_Search_and_Retrieval.ipynb
    │   ├── boto3/
    │   │   ├── 00_Tutorial_How-To.ipynb
    │   │   ├── 01_Basic_Prompt_Structure.ipynb
    │   │   ├── 02_Being_Clear_and_Direct.ipynb
    │   │   ├── 03_Assigning_Roles_Role_Prompting.ipynb
    │   │   ├── 04_Separating_Data_and_Instructions.ipynb
    │   │   ├── 05_Formatting_Output_and_Speaking_for_Claude.ipynb
    │   │   ├── 06_Precognition_Thinking_Step_by_Step.ipynb
    │   │   ├── 07_Using_Examples_Few-Shot_Prompting.ipynb
    │   │   ├── 10_1_Appendix_Chaining_Prompts.ipynb
    │   │   ├── 10_2_Appendix_Tool_Use.ipynb
    │   │   ├── 10_3_Appendix_Empirical_Performance_Eval.ipynb
    │   │   └── 10_4_Appendix_Search_and_Retrieval.ipynb
    │   ├── cloudformation/
    │   │   └── workshop-v1-final-cfn.yml
    │   └── utils/
    │       ├── __init__.py
    │       └── hints.py
    └── Anthropic 1P/
        ├── 00_Tutorial_How-To.ipynb
        ├── 01_Basic_Prompt_Structure.ipynb
        ├── 02_Being_Clear_and_Direct.ipynb
        ├── 03_Assigning_Roles_Role_Prompting.ipynb
        ├── 04_Separating_Data_and_Instructions.ipynb
        ├── 05_Formatting_Output_and_Speaking_for_Claude.ipynb
        ├── 06_Precognition_Thinking_Step_by_Step.ipynb
        ├── 07_Using_Examples_Few-Shot_Prompting.ipynb
        ├── 10.1_Appendix_Chaining Prompts.ipynb
        ├── 10.2_Appendix_Tool Use.ipynb
        ├── 10.3_Appendix_Search & Retrieval.ipynb
        └── hints.py

================================================
FILE: README.md
================================================
# Welcome to Anthropic's Prompt Engineering Interactive Tutorial

## Course introduction and goals

This course is intended to provide you with a comprehensive step-by-step understanding of how to engineer optimal prompts within Claude.

**After completing this course, you will be able to**:
- Master the basic structure of a good prompt 
- Recognize common failure modes and learn the '80/20' techniques to address them
- Understand Claude's strengths and weaknesses
- Build strong prompts from scratch for common use cases

## Course structure and content

This course is structured to allow you many chances to practice writing and troubleshooting prompts yourself. The course is broken up into **9 chapters with accompanying exercises**, as well as an appendix of even more advanced methods. It is intended for you to **work through the course in chapter order**. 

**Each lesson has an "Example Playground" area** at the bottom where you are free to experiment with the examples in the lesson and see for yourself how changing prompts can change Claude's responses. There is also an [answer key](https://docs.google.com/spreadsheets/d/1jIxjzUWG-6xBVIa2ay6yDpLyeuOh_hR_ZB75a47KX_E/edit?usp=sharing).

Note: This tutorial uses our smallest, fastest, and cheapest model, Claude 3 Haiku. Anthropic has [two other models](https://docs.anthropic.com/claude/docs/models-overview), Claude 3 Sonnet and Claude 3 Opus, which are more intelligent than Haiku, with Opus being the most intelligent.

*This tutorial also exists on [Google Sheets using Anthropic's Claude for Sheets extension](https://docs.google.com/spreadsheets/d/19jzLgRruG9kjUQNKtCg1ZjdD6l6weA6qRXG5zLIAhC8/edit?usp=sharing). We recommend using that version as it is more user friendly.*

When you are ready to begin, go to `01_Basic Prompt Structure` to proceed.

## Table of Contents

Each chapter consists of a lesson and a set of exercises.

### Beginner
- **Chapter 1:** Basic Prompt Structure

- **Chapter 2:** Being Clear and Direct  

- **Chapter 3:** Assigning Roles

### Intermediate 
- **Chapter 4:** Separating Data from Instructions

- **Chapter 5:** Formatting Output & Speaking for Claude

- **Chapter 6:** Precognition (Thinking Step by Step)

- **Chapter 7:** Using Examples

### Advanced
- **Chapter 8:** Avoiding Hallucinations

- **Chapter 9:** Building Complex Prompts (Industry Use Cases)
  - Complex Prompts from Scratch - Chatbot
  - Complex Prompts for Legal Services
  - **Exercise:** Complex Prompts for Financial Services
  - **Exercise:** Complex Prompts for Coding
  - Congratulations & Next Steps

- **Appendix:** Beyond Standard Prompting
  - Chaining Prompts
  - Tool Use
  - Search & Retrieval


================================================
FILE: AmazonBedrock/README.md
================================================
# Welcome to Anthropic's Prompt Engineering Interactive Tutorial - Bedrock Edition

## Course introduction and goals

This course is intended to provide you with a comprehensive step-by-step understanding of how to engineer optimal prompts within Claude, using Bedrock.

**After completing this course, you will be able to**:
- Master the basic structure of a good prompt 
- Recognize common failure modes and learn the '80/20' techniques to address them
- Understand Claude's strengths and weaknesses
- Build strong prompts from scratch for common use cases

## Course structure and content

This course is structured to allow you many chances to practice writing and troubleshooting prompts yourself. The course is broken up into **9 chapters with accompanying exercises**, as well as an appendix of even more advanced methods. It is intended for you to **work through the course in chapter order**. 

**Each lesson has an "Example Playground" area** at the bottom where you are free to experiment with the examples in the lesson and see for yourself how changing prompts can change Claude's responses. There is also an [answer key](https://docs.google.com/spreadsheets/d/1jIxjzUWG-6xBVIa2ay6yDpLyeuOh_hR_ZB75a47KX_E/edit?usp=sharing). While this answer key is structured for 1P API requests, the solutions are the same.

Note: This tutorial uses our smallest, fastest, and cheapest model, Claude 3 Haiku. Anthropic has [two other models](https://docs.anthropic.com/claude/docs/models-overview), Claude 3 Sonnet and Claude 3 Opus, which are more intelligent than Haiku, with Opus being the most intelligent.

When you are ready to begin, go to `01_Basic Prompt Structure` to proceed.

## Table of Contents

Each chapter consists of a lesson and a set of exercises.

### Beginner
- **Chapter 1:** Basic Prompt Structure

- **Chapter 2:** Being Clear and Direct  

- **Chapter 3:** Assigning Roles

### Intermediate 
- **Chapter 4:** Separating Data from Instructions

- **Chapter 5:** Formatting Output & Speaking for Claude

- **Chapter 6:** Precognition (Thinking Step by Step)

- **Chapter 7:** Using Examples

### Advanced
- **Chapter 8:** Avoiding Hallucinations

- **Chapter 9:** Building Complex Prompts (Industry Use Cases)
  - Complex Prompts from Scratch - Chatbot
  - Complex Prompts for Legal Services
  - **Exercise:** Complex Prompts for Financial Services
  - **Exercise:** Complex Prompts for Coding
  - Congratulations & Next Steps

- **Appendix:** Beyond Standard Prompting
  - Chaining Prompts
  - Tool Use
  - Empriical Performance Evaluations
  - Search & Retrieval


================================================
FILE: AmazonBedrock/requirements.txt
================================================
awscli==1.32.74
boto3==1.34.74
botocore==1.34.74
anthropic==0.21.3
pickleshare==0.7.5



================================================
FILE: AmazonBedrock/anthropic/00_Tutorial_How-To.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Tutorial How-To

This tutorial requires this initial notebook to be run first so that the requirements and environment variables are stored for all notebooks in the workshop
"""

"""
## How to get started

1. Clone this repository to your local machine.

2. Install the required dependencies by running the following command:
 
"""

%pip install -qU pip
%pip install -qr ../requirements.txt
# Output:
#   Note: you may need to restart the kernel to use updated packages.

#   Note: you may need to restart the kernel to use updated packages.


"""
3. Restart the kernel after installing dependencies
"""

# restart kernel
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

"""
---

## Usage Notes & Tips 💡

- This course uses Claude 3 Haiku with temperature 0. We will talk more about temperature later in the course. For now, it's enough to understand that these settings yield more deterministic results. All prompt engineering techniques in this course also apply to previous generation legacy Claude models such as Claude 2 and Claude Instant 1.2.

- You can use `Shift + Enter` to execute the cell and move to the next one.

- When you reach the bottom of a tutorial page, navigate to the next numbered file in the folder, or to the next numbered folder if you're finished with the content within that chapter file.

### The Anthropic SDK & the Messages API
We will be using the [Anthropic python SDK](https://docs.anthropic.com/claude/reference/claude-on-amazon-bedrock) and the [Messages API](https://docs.anthropic.com/claude/reference/messages_post) throughout this tutorial.

Below is an example of what running a prompt will look like in this tutorial.
"""

"""
First, we set and store the model name and region.
"""

import boto3
session = boto3.Session() # create a boto3 session to dynamically get and set the region name
AWS_REGION = session.region_name
print("AWS Region:", AWS_REGION)
MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"

%store MODEL_NAME
%store AWS_REGION

"""
Then, we create `get_completion`, which is a helper function that sends a prompt to Claude and returns Claude's generated response. Run that cell now.
"""

from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

"""
Now we will write out an example prompt for Claude and print Claude's output by running our `get_completion` helper function. Running the cell below will print out a response from Claude beneath it.

Feel free to play around with the prompt string to elicit different responses from Claude.
"""

# Prompt
prompt = "Hello, Claude!"

# Get Claude's response
print(get_completion(prompt))

"""
The `MODEL_NAME` and `AWS_REGION` variables defined earlier will be used throughout the tutorial. Just make sure to run the cells for each tutorial page from top to bottom.
"""



================================================
FILE: AmazonBedrock/anthropic/01_Basic_Prompt_Structure.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 1: Basic Prompt Structure

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

Anthropic offers two APIs, the legacy [Text Completions API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html) and the current [Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html). For this tutorial, we will be exclusively using the Messages API.

At minimum, a call to Claude using the Messages API requires the following parameters:
- `model`: the [API model name](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns) of the model that you intend to call

- `max_tokens`: the maximum number of tokens to generate before stopping. Note that Claude may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate. Furthermore, this is a *hard* stop, meaning that it may cause Claude to stop generating mid-word or mid-sentence.

- `messages`: an array of input messages. Our models are trained to operate on alternating `user` and `assistant` conversational turns. When creating a new `Message`, you specify the prior conversational turns with the messages parameter, and the model then generates the next `Message` in the conversation.
  - Each input message must be an object with a `role` and `content`. You can specify a single `user`-role message, or you can include multiple `user` and `assistant` messages (they must alternate, if so). The first message must always use the user `role`.

There are also optional parameters, such as:
- `system`: the system prompt - more on this below.
  
- `temperature`: the degree of variability in Claude's response. For these lessons and exercises, we have set `temperature` to 0.

For a complete list of all API parameters, visit our [API documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html).
"""

"""
### Examples

Let's take a look at how Claude responds to some correctly-formatted prompts. For each of the following cells, run the cell (`shift+enter`), and Claude's response will appear below the block.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now let's take a look at some prompts that do not include the correct Messages API formatting. For these malformatted prompts, the Messages API returns an error.

First, we have an example of a Messages API call that lacks `role` and `content` fields in the `messages` array.
"""

"""
> ⚠️ **Warning:** Due to the incorrect formatting of the messages parameter in the prompt, the following cell will return an error. This is expected behavior.
"""

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"Hi Claude, how are you?"}
        ]
    )

# Print Claude's response
print(response[0].text)

"""
Here's a prompt that fails to alternate between the `user` and `assistant` roles.
"""

"""
> ⚠️ **Warning:** Due to the lack of alternation between `user` and `assistant` roles, Claude will return an error message. This is expected behavior.
"""

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ]
    )

# Print Claude's response
print(response[0].text)

"""
`user` and `assistant` messages **MUST alternate**, and messages **MUST start with a `user` turn**. You can have multiple `user` & `assistant` pairs in a prompt (as if simulating a multi-turn conversation). You can also put words into a terminal `assistant` message for Claude to continue from where you left off (more on that in later chapters).

#### System Prompts

You can also use **system prompts**. A system prompt is a way to **provide context, instructions, and guidelines to Claude** before presenting it with a question or task in the "User" turn. 

Structurally, system prompts exist separately from the list of `user` & `assistant` messages, and thus belong in a separate `system` parameter (take a look at the structure of the `get_completion` helper function in the [Setup](#setup) section of the notebook). 

Within this tutorial, wherever we might utilize a system prompt, we have provided you a `system` field in your completions function. Should you not want to use a system prompt, simply set the `SYSTEM_PROMPT` variable to an empty string.
"""

"""
#### System Prompt Example
"""

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
Why use a system prompt? A **well-written system prompt can improve Claude's performance** in a variety of ways, such as increasing Claude's ability to follow rules and instructions. For more information, visit our documentation on [how to use system prompts](https://docs.anthropic.com/claude/docs/how-to-use-system-prompts) with Claude.

Now we'll dive into some exercises. If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 1.1 - Counting to Three](#exercise-11---counting-to-three)
- [Exercise 1.2 - System Prompt](#exercise-12---system-prompt)
"""

"""
### Exercise 1.1 - Counting to Three
Using proper `user` / `assistant` formatting, edit the `PROMPT` below to get Claude to **count to three.** The output will also indicate whether your solution is correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    pattern = re.compile(r'^(?=.*1)(?=.*2)(?=.*3).*$', re.DOTALL)
    return bool(pattern.match(text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_1_1_hint)

"""
### Exercise 1.2 - System Prompt

Modify the `SYSTEM_PROMPT` to make Claude respond like it's a 3 year old child.
"""

# System prompt - this is the only field you should change
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "How big is the sky?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search(r"giggles", text) or re.search(r"soo", text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_1_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"Hi Claude, how are you?"}
        ]
    )

# Print Claude's response
print(response[0].text)

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ]
    )

# Print Claude's response
print(response[0].text)

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: AmazonBedrock/anthropic/02_Being_Clear_and_Direct.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 2: Being Clear and Direct

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

**Claude responds best to clear and direct instructions.**

Think of Claude like any other human that is new to the job. **Claude has no context** on what to do aside from what you literally tell it. Just as when you instruct a human for the first time on a task, the more you explain exactly what you want in a straightforward manner to Claude, the better and more accurate Claude's response will be."				
				
When in doubt, follow the **Golden Rule of Clear Prompting**:
- Show your prompt to a colleague or friend and have them follow the instructions themselves to see if they can produce the result you want. If they're confused, Claude's confused.				
"""

"""
### Examples

Let's take a task like writing poetry. (Ignore any syllable mismatch - LLMs aren't great at counting syllables yet.)
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

"""
This haiku is nice enough, but users may want Claude to go directly into the poem without the "Here is a haiku" preamble.

How do we achieve that? We **ask for it**!
"""

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

"""
Here's another example. Let's ask Claude who's the best basketball player of all time. You can see below that while Claude lists a few names, **it doesn't respond with a definitive "best"**.
"""

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Can we get Claude to make up its mind and decide on a best player? Yes! Just ask!
"""

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 2.1 - Spanish](#exercise-21---spanish)
- [Exercise 2.2 - One Player Only](#exercise-22---one-player-only)
- [Exercise 2.3 - Write a Story](#exercise-23---write-a-story)
"""

"""
### Exercise 2.1 - Spanish
Modify the `SYSTEM_PROMPT` to make Claude output its answer in Spanish.
"""

# System prompt - this is the only field you should chnage
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "Hello Claude, how are you?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return "hola" in text.lower()

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_1_hint)

"""
### Exercise 2.2 - One Player Only

Modify the `PROMPT` so that Claude doesn't equivocate at all and responds with **ONLY** the name of one specific player, with **no other words or punctuation**. 
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return text == "Michael Jordan"

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_2_hint)

"""
### Exercise 2.3 - Write a Story

Modify the `PROMPT` so that Claude responds with as long a response as you can muster. If your answer is **over 800 words**, Claude's response will be graded as correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    trimmed = text.strip()
    words = len(trimmed.split())
    return words >= 800

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/anthropic/03_Assigning_Roles_Role_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 3: Assigning Roles (Role Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

Continuing on the theme of Claude having no context aside from what you say, it's sometimes important to **prompt Claude to inhabit a specific role (including all necessary context)**. This is also known as role prompting. The more detail to the role context, the better.

**Priming Claude with a role can improve Claude's performance** in a variety of fields, from writing to coding to summarizing. It's like how humans can sometimes be helped when told to "think like a ______". Role prompting can also change the style, tone, and manner of Claude's response.

**Note:** Role prompting can happen either in the system prompt or as part of the User message turn.
"""

"""
### Examples

In the example below, we see that without role prompting, Claude provides a **straightforward and non-stylized answer** when asked to give a single sentence perspective on skateboarding.

However, when we prime Claude to inhabit the role of a cat, Claude's perspective changes, and thus **Claude's response tone, style, content adapts to the new role**. 

**Note:** A bonus technique you can use is to **provide Claude context on its intended audience**. Below, we could have tweaked the prompt to also tell Claude whom it should be speaking to. "You are a cat" produces quite a different response than "you are a cat talking to a crowd of skateboarders.

Here is the prompt without role prompting in the system prompt:
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Here is the same user question, except with role prompting.
"""

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
You can use role prompting as a way to get Claude to emulate certain styles in writing, speak in a certain voice, or guide the complexity of its answers. **Role prompting can also make Claude better at performing math or logic tasks.**

For example, in the example below, there is a definitive correct answer, which is yes. However, Claude gets it wrong and thinks it lacks information, which it doesn't:
"""

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now, what if we **prime Claude to act as a logic bot**? How will that change Claude's answer? 

It turns out that with this new role assignment, Claude gets it right. (Although notably not for all the right reasons)
"""

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Note:** What you'll learn throughout this course is that there are **many prompt engineering techniques you can use to derive similar results**. Which techniques you use is up to you and your preference! We encourage you to **experiment to find your own prompt engineering style**.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 3.1 - Math Correction](#exercise-31---math-correction)
"""

"""
### Exercise 3.1 - Math Correction
In some instances, **Claude may struggle with mathematics**, even simple mathematics. Below, Claude incorrectly assesses the math problem as correctly solved, even though there's an obvious arithmetic mistake in the second step. Note that Claude actually catches the mistake when going through step-by-step, but doesn't jump to the conclusion that the overall solution is wrong.

Modify the `PROMPT` and / or the `SYSTEM_PROMPT` to make Claude grade the solution as `incorrectly` solved, rather than correctly solved. 

"""

# System prompt - if you don't want to use a system prompt, you can leave this variable set to an empty string
SYSTEM_PROMPT = ""

# Prompt
PROMPT = """Is this equation solved correctly below?

2x - 3 = 9
2x = 6
x = 3"""

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    if "incorrect" in text or "not correct" in text.lower():
        return True
    else:
        return False

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_3_1_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: AmazonBedrock/anthropic/04_Separating_Data_and_Instructions.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 4: Separating Data and Instructions

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

Oftentimes, we don't want to write full prompts, but instead want **prompt templates that can be modified later with additional input data before submitting to Claude**. This might come in handy if you want Claude to do the same thing every time, but the data that Claude uses for its task might be different each time. 

Luckily, we can do this pretty easily by **separating the fixed skeleton of the prompt from variable user input, then substituting the user input into the prompt** before sending the full prompt to Claude. 

Below, we'll walk step by step through how to write a substitutable prompt template, as well as how to substitute in user input.
"""

"""
### Examples

In this first example, we're asking Claude to act as an animal noise generator. Notice that the full prompt submitted to Claude is just the `PROMPT_TEMPLATE` substituted with the input (in this case, "Cow"). Notice that the word "Cow" replaces the `ANIMAL` placeholder via an f-string when we print out the full prompt.

**Note:** You don't have to call your placeholder variable anything in particular in practice. We called it `ANIMAL` in this example, but just as easily, we could have called it `CREATURE` or `A` (although it's generally good to have your variable names be specific and relevant so that your prompt template is easy to understand even without the substitution, just for user parseability). Just make sure that whatever you name your variable is what you use for the prompt template f-string.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why would we want to separate and substitute inputs like this? Well, **prompt templates simplify repetitive tasks**. Let's say you build a prompt structure that invites third party users to submit content to the prompt (in this case the animal whose sound they want to generate). These third party users don't have to write or even see the full prompt. All they have to do is fill in variables.

We do this substitution here using variables and f-strings, but you can also do it with the format() method.

**Note:** Prompt templates can have as many variables as desired!
"""

"""
When introducing substitution variables like this, it is very important to **make sure Claude knows where variables start and end** (vs. instructions or task descriptions). Let's look at an example where there is no separation between the instructions and the substitution variable.

To our human eyes, it is very clear where the variable begins and ends in the prompt template below. However, in the fully substituted prompt, that delineation becomes unclear.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Here, **Claude thinks "Yo Claude" is part of the email it's supposed to rewrite**! You can tell because it begins its rewrite with "Dear Claude". To the human eye, it's clear, particularly in the prompt template where the email begins and ends, but it becomes much less clear in the prompt after substitution.
"""

"""
How do we solve this? **Wrap the input in XML tags**! We did this below, and as you can see, there's no more "Dear Claude" in the output.

[XML tags](https://docs.anthropic.com/claude/docs/use-xml-tags) are angle-bracket tags like `<tag></tag>`. They come in pairs and consist of an opening tag, such as `<tag>`, and a closing tag marked by a `/`, such as `</tag>`. XML tags are used to wrap around content, like this: `<tag>content</tag>`.

**Note:** While Claude can recognize and work with a wide range of separators and delimeters, we recommend that you **use specifically XML tags as separators** for Claude, as Claude was trained specifically to recognize XML tags as a prompt organizing mechanism. Outside of function calling, **there are no special sauce XML tags that Claude has been trained on that you should use to maximally boost your performance**. We have purposefully made Claude very malleable and customizable this way.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Let's see another example of how XML tags can help us. 

In the following prompt, **Claude incorrectly interprets what part of the prompt is the instruction vs. the input**. It incorrectly considers `Each is about an animal, like rabbits` to be part of the list due to the formatting, when the user (the one filling out the `SENTENCES` variable) presumably did not want that.
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
To fix this, we just need to **surround the user input sentences in XML tags**. This shows Claude where the input data begins and ends despite the misleading hyphen before `Each is about an animal, like rabbits.`
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
**Note:** In the incorrect version of the "Each is about an animal" prompt, we had to include the hyphen to get Claude to respond incorrectly in the way we wanted to for this example. This is an important lesson about prompting: **small details matter**! It's always worth it to **scrub your prompts for typos and grammatical errors**. Claude is sensitive to patterns (in its early years, before finetuning, it was a raw text-prediction tool), and it's more likely to make mistakes when you make mistakes, smarter when you sound smart, sillier when you sound silly, and so on.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 4.1 - Haiku Topic](#exercise-41---haiku-topic)
- [Exercise 4.2 - Dog Question with Typos](#exercise-42---dog-question-with-typos)
- [Exercise 4.3 - Dog Question Part 2](#exercise-42---dog-question-part-2)
"""

"""
### Exercise 4.1 - Haiku Topic
Modify the `PROMPT` so that it's a template that will take in a variable called `TOPIC` and output a haiku about the topic. This exercise is just meant to test your understanding of the variable templating structure with f-strings.
"""

# Variable content
TOPIC = "Pigs"

# Prompt template with a placeholder for the variable content
PROMPT = f""

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("pigs", text.lower()) and re.search("haiku", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_1_hint)

"""
### Exercise 4.2 - Dog Question with Typos
Fix the `PROMPT` by adding XML tags so that Claude produces the right answer. 

Try not to change anything else about the prompt. The messy and mistake-ridden writing is intentional, so you can see how Claude reacts to such mistakes.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_2_hint)

"""
### Exercise 4.3 - Dog Question Part 2
Fix the `PROMPT` **WITHOUT** adding XML tags. Instead, remove only one or two words from the prompt.

Just as with the above exercises, try not to change anything else about the prompt. This will show you what kind of language Claude can parse and understand.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/anthropic/05_Formatting_Output_and_Speaking_for_Claude.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 5: Formatting Output and Speaking for Claude

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

**Claude can format its output in a wide variety of ways**. You just need to ask for it to do so!

One of these ways is by using XML tags to separate out the response from any other superfluous text. You've already learned that you can use XML tags to make your prompt clearer and more parseable to Claude. It turns out, you can also ask Claude to **use XML tags to make its output clearer and more easily understandable** to humans.
"""

"""
### Examples

Remember the 'poem preamble problem' we solved in Chapter 2 by asking Claude to skip the preamble entirely? It turns out we can also achieve a similar outcome by **telling Claude to put the poem in XML tags**.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why is this something we'd want to do? Well, having the output in **XML tags allows the end user to reliably get the poem and only the poem by writing a short program to extract the content between XML tags**.

An extension of this technique is to **put the first XML tag in the `assistant` turn. When you put text in the `assistant` turn, you're basically telling Claude that Claude has already said something, and that it should continue from that point onward. This technique is called "speaking for Claude" or "prefilling Claude's response."

Below, we've done this with the first `<haiku>` XML tag. Notice how Claude continues directly from where we left off.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Claude also excels at using other output formatting styles, notably `JSON`. If you want to enforce JSON output (not deterministically, but close to it), you can also prefill Claude's response with the opening bracket, `{`}.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Below is an example of **multiple input variables in the same prompt AND output formatting specification, all done using XML tags**.
"""

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
#### Bonus lesson

If you are calling Claude through the API, you can pass the closing XML tag to the `stop_sequences` parameter to get Claude to stop sampling once it emits your desired tag. This can save money and time-to-last-token by eliminating Claude's concluding remarks after it's already given you the answer you care about.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 5.1 - Steph Curry GOAT](#exercise-51---steph-curry-goat)
- [Exercise 5.2 - Two Haikus](#exercise-52---two-haikus)
- [Exercise 5.3 - Two Haikus, Two Animals](#exercise-53---two-haikus-two-animals)
"""

"""
### Exercise 5.1 - Steph Curry GOAT
Forced to make a choice, Claude designates Michael Jordan as the best basketball player of all time. Can we get Claude to pick someone else?

Change the `PREFILL` variable to **compell Claude to make a detailed argument that the best basketball player of all time is Stephen Curry**. Try not to change anything except `PREFILL` as that is the focus of this exercise.
"""

# Prompt template with a placeholder for the variable content
PROMPT = f"Who is the best basketball player of all time? Please choose one specific player."

# Prefill for Claude's response
PREFILL = ""

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("Warrior", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_1_hint)

"""
### Exercise 5.2 - Two Haikus
Modify the `PROMPT` below using XML tags so that Claude writes two haikus about the animal instead of just one. It should be clear where one poem ends and the other begins.
"""

# Variable content
ANIMAL = "cats"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(
        (re.search("cat", text.lower()) and re.search("<haiku>", text))
        and (text.count("\n") + 1) > 5
    )

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_2_hint)

"""
### Exercise 5.3 - Two Haikus, Two Animals
Modify the `PROMPT` below so that **Claude produces two haikus about two different animals**. Use `{ANIMAL1}` as a stand-in for the first substitution, and `{ANIMAL2}` as a stand-in for the second substitution.
"""

# First input variable
ANIMAL1 = "Cat"

# Second input variable
ANIMAL2 = "Dog"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL1}. Put it in <haiku> tags."

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("tail", text.lower()) and re.search("cat", text.lower()) and re.search("<haiku>", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: AmazonBedrock/anthropic/06_Precognition_Thinking_Step_by_Step.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 6: Precognition (Thinking Step by Step)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

If someone woke you up and immediately started asking you several complicated questions that you had to respond to right away, how would you do? Probably not as good as if you were given time to **think through your answer first**. 

Guess what? Claude is the same way.

**Giving Claude time to think step by step sometimes makes Claude more accurate**, particularly for complex tasks. However, **thinking only counts when it's out loud**. You cannot ask Claude to think but output only the answer - in this case, no thinking has actually occurred.
"""

"""
### Examples

In the prompt below, it's clear to a human reader that the second sentence belies the first. But **Claude takes the word "unrelated" too literally**.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
To improve Claude's response, let's **allow Claude to think things out first before answering**. We do that by literally spelling out the steps that Claude should take in order to process and think through its task. Along with a dash of role prompting, this empowers Claude to understand the review more deeply.
"""

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Claude is sometimes sensitive to ordering**. This example is on the frontier of Claude's ability to understand nuanced text, and when we swap the order of the arguments from the previous example so that negative is first and positive is second, this changes Claude's overall assessment to positive.

In most situations (but not all, confusingly enough), **Claude is more likely to choose the second of two options**, possibly because in its training data from the web, second options were more likely to be correct.
"""

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
**Letting Claude think can shift Claude's answer from incorrect to correct**. It's that simple in many cases where Claude makes mistakes!

Let's go through an example where Claude's answer is incorrect to see how asking Claude to think can fix that.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

"""
Let's fix this by asking Claude to think step by step, this time in `<brainstorm>` tags.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 6.1 - Classifying Emails](#exercise-61---classifying-emails)
- [Exercise 6.2 - Email Classification Formatting](#exercise-62---email-classification-formatting)
"""

"""
### Exercise 6.1 - Classifying Emails
In this exercise, we'll be instructing Claude to sort emails into the following categories:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)

For the first part of the exercise, change the `PROMPT` to **make Claude output the correct classification and ONLY the classification**. Your answer needs to **include the letter (A - D) of the correct choice, with the parentheses, as well as the name of the category**.

Refer to the comments beside each email in the `EMAILS` list to know which category that email should be classified under.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "A\) P",
    "B": "B\) B",
    "C": "C\) B",
    "D": "D\) O"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_6_1_hint)

"""
Still stuck? Run the cell below for an example solution.						
"""

print(hints.exercise_6_1_solution)

"""
### Exercise 6.2 - Email Classification Formatting
In this exercise, we're going to refine the output of the above prompt to yield an answer formatted exactly how we want it. 

Use your favorite output formatting technique to make Claude wrap JUST the letter of the correct classification in `<answer></answer>` tags. For instance, the answer to the first email should contain the exact string `<answer>B</answer>`.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "<answer>A</answer>",
    "B": "<answer>B</answer>",
    "C": "<answer>C</answer>",
    "D": "<answer>D</answer>"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_6_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/anthropic/07_Using_Examples _Few-Shot_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 7: Using Examples (Few-Shot Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ],
        system=system
    )
    return message.content[0].text

"""
---

## Lesson

**Giving Claude examples of how you want it to behave (or how you want it not to behave) is extremely effective** for:
- Getting the right answer
- Getting the answer in the right format

This sort of prompting is also called "**few shot prompting**". You might also encounter the phrase "zero-shot" or "n-shot" or "one-shot". The number of "shots" refers to how many examples are used within the prompt.
"""

"""
### Examples

Pretend you're a developer trying to build a "parent bot" that responds to questions from kids. **Claude's default response is quite formal and robotic**. This is going to break a child's heart.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

"""
You could take the time to describe your desired tone, but it's much easier just to **give Claude a few examples of ideal responses**.
"""

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

"""
In the following formatting example, we could walk Claude step by step through a set of formatting instructions on how to extract names and professions and then format them exactly the way we want, or we could just **provide Claude with some correctly-formatted examples and Claude can extrapolate from there**. Note the `<individuals>` in the `assistant` turn to start Claude off on the right foot.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 7.1 - Email Formatting via Examples](#exercise-71---email-formatting-via-examples)
"""

"""
### Exercise 7.1 - Email Formatting via Examples
We're going to redo Exercise 6.2, but this time, we're going to edit the `PROMPT` to use "few-shot" examples of emails + proper classification (and formatting) to get Claude to output the correct answer. We want the *last* letter of Claude's output to be the letter of the category.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.

Remember that these are the categories for the emails:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)								
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(ans, response[-1])) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_7_1_hint)

"""
Still stuck? Run the cell below for an example solution.
"""

print(hints.exercise_7_1_solution)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: AmazonBedrock/anthropic/10_1_Appendix_Chaining_Prompts.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.1: Chaining Prompts

- [Lesson](#lesson)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

%store -r MODEL_NAME
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

def get_completion(messages, system_prompt=''):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=messages,
        system=system_prompt
    )
    return message.content[0].text

"""
---

## Lesson

The saying goes, "Writing is rewriting." It turns out, **Claude can often improve the accuracy of its response when asked to do so**!

There are many ways to prompt Claude to "think again". The ways that feel natural to ask a human to double check their work will also generally work for Claude. (Check out our [prompt chaining documentation](https://docs.anthropic.com/claude/docs/chain-prompts) for further examples of when and how to use prompt chaining.)
"""

"""
### Examples

In this example, we ask Claude to come up with ten words... but one or more of them isn't a real word.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
**Asking Claude to make its answer more accurate** fixes the error! 

Below, we've pulled down Claude's incorrect response from above and added another turn to the conversation asking Claude to fix its previous answer.
"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
But is Claude revising its answer just because we told it to? What if we start off with a correct answer already? Will Claude lose its confidence? Here, we've placed a correct response in the place of `first_response` and asked it to double check again.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
You may notice that if you generate a respnse from the above block a few times, Claude leaves the words as is most of the time, but still occasionally changes the words even though they're all already correct. What can we do to mitigate this? Per Chapter 8, we can give Claude an out! Let's try this one more time.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Try generating responses from the above code a few times to see that Claude is much better at sticking to its guns now.

You can also use prompt chaining to **ask Claude to make its responses better**. Below, we asked Claude to first write a story, and then improve the story it wrote. Your personal tastes may vary, but many might agree that Claude's second version is better.

First, let's generate Claude's first version of the story.
"""

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
Now let's have Claude improve on its first draft.
"""

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
This form of substitution is very powerful. We've been using substitution placeholders to pass in lists, words, Claude's former responses, and so on. You can also **use substitution to do what we call "function calling," which is asking Claude to perform some function, and then taking the results of that function and asking Claude to do even more afterward with the results**. It works like any other substitution. More on this in the next appendix.

Below is one more example of taking the results of one call to Claude and plugging it into another, longer call. Let's start with the first prompt (which includes prefilling Claude's response this time).
"""

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

"""
Let's pass this list of names into another prompt.
"""

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Now that you've learned about prompt chaining, head over to Appendix 10.2 to learn how to implement function calling using prompt chaining.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))



================================================
FILE: AmazonBedrock/anthropic/10_2_Appendix_Tool_Use.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.2: Tool Use

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

%pip install anthropic --quiet

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Import python's built-in regular expression library
import re
from anthropic import AnthropicBedrock

# Override the MODEL_NAME variable in the IPython store to use Sonnet instead of the Haiku model
MODEL_NAME='anthropic.claude-3-sonnet-20240229-v1:0'
%store -r AWS_REGION

client = AnthropicBedrock(aws_region=AWS_REGION)

# Rewrittten to call Claude 3 Sonnet, which is generally better at tool use, and include stop_sequences
def get_completion(messages, system_prompt="", prefill="",stop_sequences=None):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=messages,
        system=system_prompt,
        stop_sequences=stop_sequences
    )
    return message.content[0].text

"""
---

## Lesson

While it might seem conceptually complex at first, tool use, a.k.a. function calling, is actually quite simple! You already know all the skills necessary to implement tool use, which is really just a combination of substitution and prompt chaining.

In previous substitution exercises, we substituted text into prompts. With tool use, we substitute tool or function results into prompts. Claude can't literally call or access tools and functions. Instead, we have Claude:
1. Output the tool name and arguments it wants to call
2. Halt any further response generation while the tool is called
3. Then we reprompt with the appended tool results
"""

"""
Function calling is useful because it expands Claude's capabilities and enables Claude to handle much more complex, multi-step tasks.
Some examples of functions you can give Claude:
- Calculator
- Word counter
- SQL database querying and data retrieval
- Weather API
"""

"""
You can get Claude to do tool use by combining these two elements:

1. A system prompt, in which we give Claude an explanation of the concept of tool use as well as a detailed descriptive list of the tools it has access to
2. The control logic with which to orchestrate and execute Claude's tool use requests
"""

"""
### Tool use roadmap

*This lesson teaches our current tool use format. However, we will be updating and improving tool use functionality in the near future, including:*
* *A more streamlined format for function definitions and calls*
* *More robust error handilgj and edge case coverage*
* *Tighter integration with the rest of our API*
* *Better reliability and performance, especially for more complex tool use tasks*
"""

"""
### Examples

To enable tool use in Claude, we start with the system prompt. In this special tool use system prompt, wet tell Claude:
* The basic premise of tool use and what it entails
* How Claude can call and use the tools it's been given
* A detailed list of tools it has access to in this specific scenario 

Here's the first part of the system prompt, explaining tool use to Claude. This part of the system prompt is generalizable across all instances of prompting Claude for tool use. The tool calling structure we're giving Claude (`<function_calls> [...] </function_calls>`) is a structure Claude has been specifically trained to use, so we recommend that you stick with this.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

"""
Here's the second part of the system prompt, which defines the exact tools Claude has access to in this specific situation. In this example, we will be giving Claude a calculator tool, which takes three parameters: two operands and an operator. 

Then we combine the two parts of the system prompt.
"""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

"""
Now we can give Claude a question that requires use of the `calculator` tool. We will use `<function_calls\>` in `stop_sequences` to detect if and when Claude calls the function.
"""

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Now, we can extract out the parameters from Claude's function call and actually run the function on Claude's behalf.

First we'll define the function's code.
"""

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

"""
Then we'll extract the parameters from Claude's function call response. If all the parameters exist, we run the calculator tool.
"""

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

"""
Now that we have a result, we have to properly format that result so that when we pass it back to Claude, Claude understands what tool that result is in relation to. There is a set format for this that Claude has been trained to recognize:
```
<function_results>
<result>
<tool_name>{TOOL_NAME}</tool_name>
<stdout>
{TOOL_RESULT}
</stdout>
</result>
</function_results>
```

Run the cell below to format the above tool result into this structure.
"""

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

"""
Now all we have to do is send this result back to Claude by appending the result to the same message chain as before, and we're good!
"""

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

"""
Congratulations on running an entire tool use chain end to end!

Now what if we give Claude a question that doesn't that doesn't require using the given tool at all?
"""

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Success! As you can see, Claude knew not to call the function when it wasn't needed.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 10.2.1 - SQL](#exercise-1021---SQL)
"""

"""
### Exercise 10.2.1 - SQL
In this exercise, you'll be writing a tool use prompt for querying and writing to the world's smallest "database". Here's the initialized database, which is really just a dictionary.
"""

db = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "products": [
        {"id": 1, "name": "Widget", "price": 9.99},
        {"id": 2, "name": "Gadget", "price": 14.99},
        {"id": 3, "name": "Doohickey", "price": 19.99}
    ]
}

"""
And here is the code for the functions that write to and from the database.
"""

def get_user(user_id):
    for user in db["users"]:
        if user["id"] == user_id:
            return user
    return None

def get_product(product_id):
    for product in db["products"]:
        if product["id"] == product_id:
            return product
    return None

def add_user(name, email):
    user_id = len(db["users"]) + 1
    user = {"id": user_id, "name": name, "email": email}
    db["users"].append(user)
    return user

def add_product(name, price):
    product_id = len(db["products"]) + 1
    product = {"id": product_id, "name": name, "price": price}
    db["products"].append(product)
    return product

"""
To solve the exercise, start by defining a system prompt like `system_prompt_tools_specific_tools` above. Make sure to include the name and description of each tool, along with the name and type and description of each parameter for each function. We've given you some starting scaffolding below.
"""

system_prompt_tools_specific_tools_sql = """
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools_sql

"""
When you're ready, you can try out your tool definition system prompt on the examples below. Just run the below cell!
"""

examples = [
    "Add a user to the database named Deborah.",
    "Add a product to the database named Thingo",
    "Tell me the name of User 2",
    "Tell me the name of Product 3"
]

for example in examples:
    message = {
        "role": "user",
        "content": example
    }

    # Get & print Claude's response
    function_calling_response = get_completion([message], system_prompt=system_prompt, stop_sequences=stop_sequences)
    print(example, "\n----------\n\n", function_calling_response, "\n*********\n*********\n*********\n\n")

"""
If you did it right, the function calling messages should call the `add_user`, `add_product`, `get_user`, and `get_product` functions correctly.

For extra credit, add some code cells and write parameter-parsing code. Then call the functions with the parameters Claude gives you to see the state of the "database" after the call.
"""

"""
❓ If you want to see a possible solution, run the cell below!
"""

print(hints.exercise_10_2_1_solution)

"""
### Congrats!

Congratulations on learning tool use and function calling! Head over to the last appendix section if you would like to learn more about search & RAG.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)



================================================
FILE: AmazonBedrock/anthropic/10_3_Appendix_Empirical_Performance_Evaluations.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Evaluating AI Models: Code, Human, and Model-Based Grading

In this notebook, we'll delve into a trio of widely-used techniques for assessing the effectiveness of AI models, like Claude v3:

1. Code-based grading
2. Human grading
3. Model-based grading

We'll illustrate each approach through examples and examine their respective advantages and limitations, when gauging AI performance.
"""

"""
## Code-Based Grading Example: Sentiment Analysis

In this example, we'll evaluate Claude's ability to classify the sentiment of movie reviews as positive or negative. We can use code to check if the model's output matches the expected sentiment.
"""

# Store the model name and AWS region for later use
MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = "us-west-2"

%store MODEL_NAME
%store AWS_REGION

# Install the Anthropic package
%pip install anthropic --quiet

# Import the AnthropicBedrock class and create a client instance
from anthropic import AnthropicBedrock

client = AnthropicBedrock(aws_region=AWS_REGION)

# Function to build the input prompt for sentiment analysis
def build_input_prompt(review):
    user_content = f"""Classify the sentiment of the following movie review as either 'positive' or 'negative' provide only one of those two choices:
    <review>{review}</review>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "review": "This movie was amazing! The acting was superb and the plot kept me engaged from start to finish.",
        "golden_answer": "positive"
    },
    {
        "review": "I was thoroughly disappointed by this film. The pacing was slow and the characters were one-dimensional.",
        "golden_answer": "negative"
    }
]

# Function to get completions from the model
def get_completion(messages):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=messages
    )
    return message.content[0].text

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["review"])) for item in eval]

# Print the outputs and golden answers
for output, question in zip(outputs, eval):
    print(f"Review: {question['review']}\nGolden Answer: {question['golden_answer']}\nOutput: {output}\n")

# Function to grade the completions
def grade_completion(output, golden_answer):
    return output.lower() == golden_answer.lower()

# Grade the completions and print the accuracy
grades = [grade_completion(output, item["golden_answer"]) for output, item in zip(outputs, eval)]
print(f"Accuracy: {sum(grades) / len(grades) * 100}%")

"""
## Human Grading Example: Essay Scoring

Some tasks, like scoring essays, are difficult to evaluate with code alone. In this case, we can provide guidelines for human graders to assess the model's output.
"""

# Function to build the input prompt for essay generation
def build_input_prompt(topic):
    user_content = f"""Write a short essay discussing the following topic:
    <topic>{topic}</topic>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "topic": "The importance of education in personal development and societal progress",
        "golden_answer": "A high-scoring essay should have a clear thesis, well-structured paragraphs, and persuasive examples discussing how education contributes to individual growth and broader societal advancement."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["topic"])) for item in eval]

# Print the outputs and golden answers
for output, item in zip(outputs, eval):
    print(f"Topic: {item['topic']}\n\nGrading Rubric:\n {item['golden_answer']}\n\nModel Output:\n{output}\n")

"""
## Model-Based Grading Examples

We can use Claude to grade its own outputs by providing the model's response and a grading rubric. This allows us to automate the evaluation of tasks that would typically require human judgment.
"""

"""
### Example 1: Summarization

In this example, we'll use Claude to assess the quality of a summary it generated. This can be useful when you need to evaluate the model's ability to capture key information from a longer text concisely and accurately. By providing a rubric that outlines the essential points that should be covered, we can automate the grading process and quickly assess the model's performance on summarization tasks.
"""

# Function to build the input prompt for summarization
def build_input_prompt(text):
    user_content = f"""Please summarize the main points of the following text:
    <text>{text}</text>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing summary quality
def build_grader_prompt(output, rubric):
    user_content = f"""Assess the quality of the following summary based on this rubric:
    <rubric>{rubric}</rubric>
    <summary>{output}</summary>
    Provide a score from 1-5, where 1 is poor and 5 is excellent."""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "text": "The Magna Carta, signed in 1215, was a pivotal document in English history. It limited the powers of the monarchy and established the principle that everyone, including the king, was subject to the law. This laid the foundation for constitutional governance and the rule of law in England and influenced legal systems worldwide.",
        "golden_answer": "A high-quality summary should concisely capture the key points: 1) The Magna Carta's significance in English history, 2) Its role in limiting monarchical power, 3) Establishing the principle of rule of law, and 4) Its influence on legal systems around the world."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["text"])) for item in eval]

# Grade the completions
grades = [get_completion(build_grader_prompt(output, item["golden_answer"])) for output, item in zip(outputs, eval)]

# Print the summary quality score
print(f"Summary quality score: {grades[0]}")

"""
### Example 2: Fact-Checking

In this example, we'll use Claude to fact-check a claim and then assess the accuracy of its fact-checking. This can be useful when you need to evaluate the model's ability to distinguish between accurate and inaccurate information. By providing a rubric that outlines the key points that should be covered in a correct fact-check, we can automate the grading process and quickly assess the model's performance on fact-checking tasks.
"""

# Function to build the input prompt for fact-checking
def build_input_prompt(claim):
    user_content = f"""Determine if the following claim is true or false:
    <claim>{claim}</claim>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing fact-check accuracy
def build_grader_prompt(output, rubric):
    user_content = f"""Based on the following rubric, assess whether the fact-check is correct:
    <rubric>{rubric}</rubric>
    <fact-check>{output}</fact-check>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "claim": "The Great Wall of China is visible from space.",
        "golden_answer": "A correct fact-check should state that this claim is false. While the Great Wall is an impressive structure, it is not visible from space with the naked eye."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["claim"])) for item in eval]

grades = []
for output, item in zip(outputs, eval):
    # Print the claim, fact-check, and rubric
    print(f"Claim: {item['claim']}\n")
    print(f"Fact-check: {output}]\n")
    print(f"Rubric: {item['golden_answer']}\n")
    
    # Grade the fact-check
    grader_prompt = build_grader_prompt(output, item["golden_answer"])
    grade = get_completion(grader_prompt)
    grades.append("correct" in grade.lower())

# Print the fact-checking accuracy
accuracy = sum(grades) / len(grades)
print(f"Fact-checking accuracy: {accuracy * 100}%")

"""
### Example 3: Tone Analysis

In this example, we'll use Claude to analyze the tone of a given text and then assess the accuracy of its analysis. This can be useful when you need to evaluate the model's ability to identify and interpret the emotional content and attitudes expressed in a piece of text. By providing a rubric that outlines the key aspects of tone that should be identified, we can automate the grading process and quickly assess the model's performance on tone analysis tasks.
"""

# Function to build the input prompt for tone analysis
def build_input_prompt(text):
    user_content = f"""Analyze the tone of the following text:
    <text>{text}</text>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing tone analysis accuracy
def build_grader_prompt(output, rubric):
    user_content = f"""Assess the accuracy of the following tone analysis based on this rubric:
    <rubric>{rubric}</rubric>
    <tone-analysis>{output}</tone-analysis>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "text": "I can't believe they canceled the event at the last minute. This is completely unacceptable and unprofessional!",
        "golden_answer": "The tone analysis should identify the text as expressing frustration, anger, and disappointment. Key words like 'can't believe', 'last minute', 'unacceptable', and 'unprofessional' indicate strong negative emotions."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["text"])) for item in eval]

# Grade the completions
grades = [get_completion(build_grader_prompt(output, item["golden_answer"])) for output, item in zip(outputs, eval)]

# Print the tone analysis quality
print(f"Tone analysis quality: {grades[0]}")

"""
These examples demonstrate how code-based, human, and model-based grading can be used to evaluate AI models like Claude on various tasks. The choice of evaluation method depends on the nature of the task and the resources available. Model-based grading offers a promising approach for automating the assessment of complex tasks that would otherwise require human judgment.
"""



================================================
FILE: AmazonBedrock/anthropic/10_4_Appendix_Search_and_Retrieval.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.3: Search & Retrieval

Did you know you can use Claude to **search through Wikipedia for you**? Claude can find and retrieve articles, at which point you can also use Claude to summarize and synthesize them, write novel content from what it found, and much more. And not just Wikipedia! You can also search over your own docs, whether stored as plain text or embedded in a vector datastore.

See our [RAG cookbook examples](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/Wikipedia/wikipedia-search-cookbook.ipynb) to learn how to supplement Claude's knowledge and improve the accuracy and relevance of Claude's responses with data retrieved from vector databases, Wikipedia, the internet, and more. There, you can also learn about how to use certain [embeddings](https://docs.anthropic.com/claude/docs/embeddings) and vector database tools.

If you are interested in learning about advanced RAG architectures using Claude, check out our [Claude 3 technical presentation slides on RAG architectures](https://docs.google.com/presentation/d/1zxkSI7lLUBrZycA-_znwqu8DDyVhHLkQGScvzaZrUns/edit#slide=id.g2c736259dac_63_782).
"""



================================================
FILE: AmazonBedrock/boto3/00_Tutorial_How-To.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Tutorial How-To

This tutorial requires this initial notebook to be run first so that the requirements and environment variables are stored for all notebooks in the workshop.
"""

"""
## How to get started

1. Clone this repository to your local machine.

2. Install the required dependencies by running the following command:
 
"""

"""
> ⚠️ **Please ignore error messages related to pip's dependency resolver.**
"""

%pip install -qU pip
%pip install -qr ../requirements.txt

"""
3. Restart the kernel after installing dependencies
"""

# restart kernel
from IPython.core.display import HTML
HTML("<script>Jupyter.notebook.kernel.restart()</script>")

"""
4. Run the notebook cells in order, following the instructions provided.
"""

"""
---

## Usage Notes & Tips 💡

- This course uses Claude 3 Haiku with temperature 0. We will talk more about temperature later in the course. For now, it's enough to understand that these settings yield more deterministic results. All prompt engineering techniques in this course also apply to previous generation legacy Claude models such as Claude 2 and Claude Instant 1.2.

- You can use `Shift + Enter` to execute the cell and move to the next one.

- When you reach the bottom of a tutorial page, navigate to the next numbered file in the folder, or to the next numbered folder if you're finished with the content within that chapter file.

### The Anthropic SDK & the Messages API
We will be using the [Anthropic python SDK](https://docs.anthropic.com/claude/reference/client-sdks) and the [Messages API](https://docs.anthropic.com/claude/reference/messages_post) throughout this tutorial. 

Below is an example of what running a prompt will look like in this tutorial. First, we create `get_completion`, which is a helper function that sends a prompt to Claude and returns Claude's generated response. Run that cell now.
"""

"""
First, we set and store the model name and region.
"""

import boto3
session = boto3.Session() # create a boto3 session to dynamically get and set the region name
AWS_REGION = session.region_name
print("AWS Region:", AWS_REGION)
MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"

%store MODEL_NAME
%store AWS_REGION

"""
Then, we create `get_completion`, which is a helper function that sends a prompt to Claude and returns Claude's generated response. Run that cell now.
"""

import boto3
import json

bedrock = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1,
            "system": ''
        }
    )
    response = bedrock.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
Now we will write out an example prompt for Claude and print Claude's output by running our `get_completion` helper function. Running the cell below will print out a response from Claude beneath it.

Feel free to play around with the prompt string to elicit different responses from Claude.
"""

# Prompt
prompt = "Hello, Claude!"

# Get Claude's response
print(get_completion(prompt))

"""
The `MODEL_NAME` and `AWS_REGION` variables defined earlier will be used throughout the tutorial. Just make sure to run the cells for each tutorial page from top to bottom.
"""



================================================
FILE: AmazonBedrock/boto3/01_Basic_Prompt_Structure.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 1: Basic Prompt Structure

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt,system=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

Anthropic offers two APIs, the legacy [Text Completions API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-text-completion.html) and the current [Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html). For this tutorial, we will be exclusively using the Messages API.

At minimum, a call to Claude using the Messages API requires the following parameters:
- `model`: the [API model name](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids.html#model-ids-arns) of the model that you intend to call

- `max_tokens`: the maximum number of tokens to generate before stopping. Note that Claude may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate. Furthermore, this is a *hard* stop, meaning that it may cause Claude to stop generating mid-word or mid-sentence.

- `messages`: an array of input messages. Our models are trained to operate on alternating `user` and `assistant` conversational turns. When creating a new `Message`, you specify the prior conversational turns with the messages parameter, and the model then generates the next `Message` in the conversation.
  - Each input message must be an object with a `role` and `content`. You can specify a single `user`-role message, or you can include multiple `user` and `assistant` messages (they must alternate, if so). The first message must always use the user `role`.

There are also optional parameters, such as:
- `system`: the system prompt - more on this below.
  
- `temperature`: the degree of variability in Claude's response. For these lessons and exercises, we have set `temperature` to 0.

For a complete list of all API parameters, visit our [API documentation](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html).
"""

"""
### Examples

Let's take a look at how Claude responds to some correctly-formatted prompts. For each of the following cells, run the cell (`shift+enter`), and Claude's response will appear below the block.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now let's take a look at some prompts that do not include the correct Messages API formatting. For these malformatted prompts, the Messages API returns an error.

First, we have an example of a Messages API call that lacks `role` and `content` fields in the `messages` array.
"""

"""
> ⚠️ **Warning:** Due to the incorrect formatting of the messages parameter in the prompt, the following cell will return an error. This is expected behavior.
"""

# Get Claude's response
body = json.dumps(
    {
        "anthropic_version": '',
        "max_tokens": 2000,
        "messages": [{"Hi Claude, how are you?"}],
        "temperature": 0.0,
        "top_p": 1,
        "system": ''
    }
)

response = client.invoke_model(body=body, modelId=MODEL_NAME)

# Print Claude's response
print(response[0].text)

"""
Here's a prompt that fails to alternate between the `user` and `assistant` roles.
"""

"""
> ⚠️ **Warning:** Due to the lack of alternation between `user` and `assistant` roles, Claude will return an error message. This is expected behavior.
"""

# Get Claude's response
body = json.dumps(
    {
        "anthropic_version": '',
        "max_tokens": 2000,
        "messages": [
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ],
        "temperature": 0.0,
        "top_p": 1,
        "system": ''
    }
)

response = client.invoke_model(body=body, modelId=MODEL_NAME)

# Print Claude's response
print(response[0].text)

"""
`user` and `assistant` messages **MUST alternate**, and messages **MUST start with a `user` turn**. You can have multiple `user` & `assistant` pairs in a prompt (as if simulating a multi-turn conversation). You can also put words into a terminal `assistant` message for Claude to continue from where you left off (more on that in later chapters).

#### System Prompts

You can also use **system prompts**. A system prompt is a way to **provide context, instructions, and guidelines to Claude** before presenting it with a question or task in the "User" turn. 

Structurally, system prompts exist separately from the list of `user` & `assistant` messages, and thus belong in a separate `system` parameter (take a look at the structure of the `get_completion` helper function in the [Setup](#setup) section of the notebook). 

Within this tutorial, wherever we might utilize a system prompt, we have provided you a `system` field in your completions function. Should you not want to use a system prompt, simply set the `SYSTEM_PROMPT` variable to an empty string.
"""

"""
#### System Prompt Example
"""

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
Why use a system prompt? A **well-written system prompt can improve Claude's performance** in a variety of ways, such as increasing Claude's ability to follow rules and instructions. For more information, visit our documentation on [how to use system prompts](https://docs.anthropic.com/claude/docs/how-to-use-system-prompts) with Claude.

Now we'll dive into some exercises. If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 1.1 - Counting to Three](#exercise-11---counting-to-three)
- [Exercise 1.2 - System Prompt](#exercise-12---system-prompt)
"""

"""
### Exercise 1.1 - Counting to Three
Using proper `user` / `assistant` formatting, edit the `PROMPT` below to get Claude to **count to three.** The output will also indicate whether your solution is correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    pattern = re.compile(r'^(?=.*1)(?=.*2)(?=.*3).*$', re.DOTALL)
    return bool(pattern.match(text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_1_1_hint)

"""
### Exercise 1.2 - System Prompt

Modify the `SYSTEM_PROMPT` to make Claude respond like it's a 3 year old child.
"""

# System prompt - this is the only field you should change
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "How big is the sky?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search(r"giggles", text) or re.search(r"soo", text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_1_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

# Get Claude's response
body = json.dumps(
    {
        "anthropic_version": '',
        "max_tokens": 2000,
        "messages": [{"Hi Claude, how are you?"}],
        "temperature": 0.0,
        "top_p": 1,
        "system": ''
    }
)

response = client.invoke_model(body=body, modelId=MODEL_NAME)

# Print Claude's response
print(response[0].text)

# Get Claude's response
body = json.dumps(
    {
        "anthropic_version": '',
        "max_tokens": 2000,
        "messages": [
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ],
        "temperature": 0.0,
        "top_p": 1,
        "system": ''
    }
)

response = client.invoke_model(body=body, modelId=MODEL_NAME)

# Print Claude's response
print(response[0].text)

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: AmazonBedrock/boto3/02_Being_Clear_and_Direct.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 2: Being Clear and Direct

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt,system=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

**Claude responds best to clear and direct instructions.**

Think of Claude like any other human that is new to the job. **Claude has no context** on what to do aside from what you literally tell it. Just as when you instruct a human for the first time on a task, the more you explain exactly what you want in a straightforward manner to Claude, the better and more accurate Claude's response will be."				
				
When in doubt, follow the **Golden Rule of Clear Prompting**:
- Show your prompt to a colleague or friend and have them follow the instructions themselves to see if they can produce the result you want. If they're confused, Claude's confused.				
"""

"""
### Examples

Let's take a task like writing poetry. (Ignore any syllable mismatch - LLMs aren't great at counting syllables yet.)
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

"""
This haiku is nice enough, but users may want Claude to go directly into the poem without the "Here is a haiku" preamble.

How do we achieve that? We **ask for it**!
"""

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

"""
Here's another example. Let's ask Claude who's the best basketball player of all time. You can see below that while Claude lists a few names, **it doesn't respond with a definitive "best"**.
"""

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Can we get Claude to make up its mind and decide on a best player? Yes! Just ask!
"""

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 2.1 - Spanish](#exercise-21---spanish)
- [Exercise 2.2 - One Player Only](#exercise-22---one-player-only)
- [Exercise 2.3 - Write a Story](#exercise-23---write-a-story)
"""

"""
### Exercise 2.1 - Spanish
Modify the `SYSTEM_PROMPT` to make Claude output its answer in Spanish.
"""

# System prompt - this is the only field you should chnage
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "Hello Claude, how are you?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return "hola" in text.lower()

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_1_hint)

"""
### Exercise 2.2 - One Player Only

Modify the `PROMPT` so that Claude doesn't equivocate at all and responds with **ONLY** the name of one specific player, with **no other words or punctuation**. 
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return text == "Michael Jordan"

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_2_hint)

"""
### Exercise 2.3 - Write a Story

Modify the `PROMPT` so that Claude responds with as long a response as you can muster. If your answer is **over 800 words**, Claude's response will be graded as correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    trimmed = text.strip()
    words = len(trimmed.split())
    return words >= 800

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_2_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/boto3/03_Assigning_Roles_Role_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 3: Assigning Roles (Role Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt,system=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

Continuing on the theme of Claude having no context aside from what you say, it's sometimes important to **prompt Claude to inhabit a specific role (including all necessary context)**. This is also known as role prompting. The more detail to the role context, the better.

**Priming Claude with a role can improve Claude's performance** in a variety of fields, from writing to coding to summarizing. It's like how humans can sometimes be helped when told to "think like a ______". Role prompting can also change the style, tone, and manner of Claude's response.

**Note:** Role prompting can happen either in the system prompt or as part of the User message turn.
"""

"""
### Examples

In the example below, we see that without role prompting, Claude provides a **straightforward and non-stylized answer** when asked to give a single sentence perspective on skateboarding.

However, when we prime Claude to inhabit the role of a cat, Claude's perspective changes, and thus **Claude's response tone, style, content adapts to the new role**. 

**Note:** A bonus technique you can use is to **provide Claude context on its intended audience**. Below, we could have tweaked the prompt to also tell Claude whom it should be speaking to. "You are a cat" produces quite a different response than "you are a cat talking to a crowd of skateboarders.

Here is the prompt without role prompting in the system prompt:
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Here is the same user question, except with role prompting.
"""

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
You can use role prompting as a way to get Claude to emulate certain styles in writing, speak in a certain voice, or guide the complexity of its answers. **Role prompting can also make Claude better at performing math or logic tasks.**

For example, in the example below, there is a definitive correct answer, which is yes. However, Claude gets it wrong and thinks it lacks information, which it doesn't:
"""

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now, what if we **prime Claude to act as a logic bot**? How will that change Claude's answer? 

It turns out that with this new role assignment, Claude gets it right. (Although notably not for all the right reasons)
"""

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Note:** What you'll learn throughout this course is that there are **many prompt engineering techniques you can use to derive similar results**. Which techniques you use is up to you and your preference! We encourage you to **experiment to find your own prompt engineering style**.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 3.1 - Math Correction](#exercise-31---math-correction)
"""

"""
### Exercise 3.1 - Math Correction
In some instances, **Claude may struggle with mathematics**, even simple mathematics. Below, Claude incorrectly assesses the math problem as correctly solved, even though there's an obvious arithmetic mistake in the second step. Note that Claude actually catches the mistake when going through step-by-step, but doesn't jump to the conclusion that the overall solution is wrong.

Modify the `PROMPT` and / or the `SYSTEM_PROMPT` to make Claude grade the solution as `incorrectly` solved, rather than correctly solved. 

"""

# System prompt - if you don't want to use a system prompt, you can leave this variable set to an empty string
SYSTEM_PROMPT = ""

# Prompt
PROMPT = """Is this equation solved correctly below?

2x - 3 = 9
2x = 6
x = 3"""

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    if "incorrect" in text or "not correct" in text.lower():
        return True
    else:
        return False

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_3_1_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: AmazonBedrock/boto3/04_Separating_Data_and_Instructions.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 4: Separating Data and Instructions

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt,system=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

Oftentimes, we don't want to write full prompts, but instead want **prompt templates that can be modified later with additional input data before submitting to Claude**. This might come in handy if you want Claude to do the same thing every time, but the data that Claude uses for its task might be different each time. 

Luckily, we can do this pretty easily by **separating the fixed skeleton of the prompt from variable user input, then substituting the user input into the prompt** before sending the full prompt to Claude. 

Below, we'll walk step by step through how to write a substitutable prompt template, as well as how to substitute in user input.
"""

"""
### Examples

In this first example, we're asking Claude to act as an animal noise generator. Notice that the full prompt submitted to Claude is just the `PROMPT_TEMPLATE` substituted with the input (in this case, "Cow"). Notice that the word "Cow" replaces the `ANIMAL` placeholder via an f-string when we print out the full prompt.

**Note:** You don't have to call your placeholder variable anything in particular in practice. We called it `ANIMAL` in this example, but just as easily, we could have called it `CREATURE` or `A` (although it's generally good to have your variable names be specific and relevant so that your prompt template is easy to understand even without the substitution, just for user parseability). Just make sure that whatever you name your variable is what you use for the prompt template f-string.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why would we want to separate and substitute inputs like this? Well, **prompt templates simplify repetitive tasks**. Let's say you build a prompt structure that invites third party users to submit content to the prompt (in this case the animal whose sound they want to generate). These third party users don't have to write or even see the full prompt. All they have to do is fill in variables.

We do this substitution here using variables and f-strings, but you can also do it with the format() method.

**Note:** Prompt templates can have as many variables as desired!
"""

"""
When introducing substitution variables like this, it is very important to **make sure Claude knows where variables start and end** (vs. instructions or task descriptions). Let's look at an example where there is no separation between the instructions and the substitution variable.

To our human eyes, it is very clear where the variable begins and ends in the prompt template below. However, in the fully substituted prompt, that delineation becomes unclear.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Here, **Claude thinks "Yo Claude" is part of the email it's supposed to rewrite**! You can tell because it begins its rewrite with "Dear Claude". To the human eye, it's clear, particularly in the prompt template where the email begins and ends, but it becomes much less clear in the prompt after substitution.
"""

"""
How do we solve this? **Wrap the input in XML tags**! We did this below, and as you can see, there's no more "Dear Claude" in the output.

[XML tags](https://docs.anthropic.com/claude/docs/use-xml-tags) are angle-bracket tags like `<tag></tag>`. They come in pairs and consist of an opening tag, such as `<tag>`, and a closing tag marked by a `/`, such as `</tag>`. XML tags are used to wrap around content, like this: `<tag>content</tag>`.

**Note:** While Claude can recognize and work with a wide range of separators and delimeters, we recommend that you **use specifically XML tags as separators** for Claude, as Claude was trained specifically to recognize XML tags as a prompt organizing mechanism. Outside of function calling, **there are no special sauce XML tags that Claude has been trained on that you should use to maximally boost your performance**. We have purposefully made Claude very malleable and customizable this way.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Let's see another example of how XML tags can help us. 

In the following prompt, **Claude incorrectly interprets what part of the prompt is the instruction vs. the input**. It incorrectly considers `Each is about an animal, like rabbits` to be part of the list due to the formatting, when the user (the one filling out the `SENTENCES` variable) presumably did not want that.
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
To fix this, we just need to **surround the user input sentences in XML tags**. This shows Claude where the input data begins and ends despite the misleading hyphen before `Each is about an animal, like rabbits.`
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
**Note:** In the incorrect version of the "Each is about an animal" prompt, we had to include the hyphen to get Claude to respond incorrectly in the way we wanted to for this example. This is an important lesson about prompting: **small details matter**! It's always worth it to **scrub your prompts for typos and grammatical errors**. Claude is sensitive to patterns (in its early years, before finetuning, it was a raw text-prediction tool), and it's more likely to make mistakes when you make mistakes, smarter when you sound smart, sillier when you sound silly, and so on.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 4.1 - Haiku Topic](#exercise-41---haiku-topic)
- [Exercise 4.2 - Dog Question with Typos](#exercise-42---dog-question-with-typos)
- [Exercise 4.3 - Dog Question Part 2](#exercise-42---dog-question-part-2)
"""

"""
### Exercise 4.1 - Haiku Topic
Modify the `PROMPT` so that it's a template that will take in a variable called `TOPIC` and output a haiku about the topic. This exercise is just meant to test your understanding of the variable templating structure with f-strings.
"""

# Variable content
TOPIC = "Pigs"

# Prompt template with a placeholder for the variable content
PROMPT = f""

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("pigs", text.lower()) and re.search("haiku", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_1_hint)

"""
### Exercise 4.2 - Dog Question with Typos
Fix the `PROMPT` by adding XML tags so that Claude produces the right answer. 

Try not to change anything else about the prompt. The messy and mistake-ridden writing is intentional, so you can see how Claude reacts to such mistakes.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_2_hint)

"""
### Exercise 4.3 - Dog Question Part 2
Fix the `PROMPT` **WITHOUT** adding XML tags. Instead, remove only one or two words from the prompt.

Just as with the above exercises, try not to change anything else about the prompt. This will show you what kind of language Claude can parse and understand.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_4_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/boto3/05_Formatting_Output_and_Speaking_for_Claude.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 5: Formatting Output and Speaking for Claude

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages":[
              {"role": "user", "content": prompt},
              {"role": "assistant", "content": prefill}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

**Claude can format its output in a wide variety of ways**. You just need to ask for it to do so!

One of these ways is by using XML tags to separate out the response from any other superfluous text. You've already learned that you can use XML tags to make your prompt clearer and more parseable to Claude. It turns out, you can also ask Claude to **use XML tags to make its output clearer and more easily understandable** to humans.
"""

"""
### Examples

Remember the 'poem preamble problem' we solved in Chapter 2 by asking Claude to skip the preamble entirely? It turns out we can also achieve a similar outcome by **telling Claude to put the poem in XML tags**.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why is this something we'd want to do? Well, having the output in **XML tags allows the end user to reliably get the poem and only the poem by writing a short program to extract the content between XML tags**.

An extension of this technique is to **put the first XML tag in the `assistant` turn. When you put text in the `assistant` turn, you're basically telling Claude that Claude has already said something, and that it should continue from that point onward. This technique is called "speaking for Claude" or "prefilling Claude's response."

Below, we've done this with the first `<haiku>` XML tag. Notice how Claude continues directly from where we left off.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Claude also excels at using other output formatting styles, notably `JSON`. If you want to enforce JSON output (not deterministically, but close to it), you can also prefill Claude's response with the opening bracket, `{`}.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Below is an example of **multiple input variables in the same prompt AND output formatting specification, all done using XML tags**.
"""

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
#### Bonus lesson

If you are calling Claude through the API, you can pass the closing XML tag to the `stop_sequences` parameter to get Claude to stop sampling once it emits your desired tag. This can save money and time-to-last-token by eliminating Claude's concluding remarks after it's already given you the answer you care about.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 5.1 - Steph Curry GOAT](#exercise-51---steph-curry-goat)
- [Exercise 5.2 - Two Haikus](#exercise-52---two-haikus)
- [Exercise 5.3 - Two Haikus, Two Animals](#exercise-53---two-haikus-two-animals)
"""

"""
### Exercise 5.1 - Steph Curry GOAT
Forced to make a choice, Claude designates Michael Jordan as the best basketball player of all time. Can we get Claude to pick someone else?

Change the `PREFILL` variable to **compell Claude to make a detailed argument that the best basketball player of all time is Stephen Curry**. Try not to change anything except `PREFILL` as that is the focus of this exercise.
"""

# Prompt template with a placeholder for the variable content
PROMPT = f"Who is the best basketball player of all time? Please choose one specific player."

# Prefill for Claude's response
PREFILL = ""

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("Warrior", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_1_hint)

"""
### Exercise 5.2 - Two Haikus
Modify the `PROMPT` below using XML tags so that Claude writes two haikus about the animal instead of just one. It should be clear where one poem ends and the other begins.
"""

# Variable content
ANIMAL = "cats"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(
        (re.search("cat", text.lower()) and re.search("<haiku>", text))
        and (text.count("\n") + 1) > 5
    )

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_2_hint)

"""
### Exercise 5.3 - Two Haikus, Two Animals
Modify the `PROMPT` below so that **Claude produces two haikus about two different animals**. Use `{ANIMAL1}` as a stand-in for the first substitution, and `{ANIMAL2}` as a stand-in for the second substitution.
"""

# First input variable
ANIMAL1 = "Cat"

# Second input variable
ANIMAL2 = "Dog"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL1}. Put it in <haiku> tags."

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("tail", text.lower()) and re.search("cat", text.lower()) and re.search("<haiku>", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_5_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: AmazonBedrock/boto3/06_Precognition_Thinking_Step_by_Step.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 6: Precognition (Thinking Step by Step)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages":[
              {"role": "user", "content": prompt},
              {"role": "assistant", "content": prefill}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

If someone woke you up and immediately started asking you several complicated questions that you had to respond to right away, how would you do? Probably not as good as if you were given time to **think through your answer first**. 

Guess what? Claude is the same way.

**Giving Claude time to think step by step sometimes makes Claude more accurate**, particularly for complex tasks. However, **thinking only counts when it's out loud**. You cannot ask Claude to think but output only the answer - in this case, no thinking has actually occurred.
"""

"""
### Examples

In the prompt below, it's clear to a human reader that the second sentence belies the first. But **Claude takes the word "unrelated" too literally**.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
To improve Claude's response, let's **allow Claude to think things out first before answering**. We do that by literally spelling out the steps that Claude should take in order to process and think through its task. Along with a dash of role prompting, this empowers Claude to understand the review more deeply.
"""

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Claude is sometimes sensitive to ordering**. This example is on the frontier of Claude's ability to understand nuanced text, and when we swap the order of the arguments from the previous example so that negative is first and positive is second, this changes Claude's overall assessment to positive.

In most situations (but not all, confusingly enough), **Claude is more likely to choose the second of two options**, possibly because in its training data from the web, second options were more likely to be correct.
"""

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
**Letting Claude think can shift Claude's answer from incorrect to correct**. It's that simple in many cases where Claude makes mistakes!

Let's go through an example where Claude's answer is incorrect to see how asking Claude to think can fix that.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

"""
Let's fix this by asking Claude to think step by step, this time in `<brainstorm>` tags.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 6.1 - Classifying Emails](#exercise-61---classifying-emails)
- [Exercise 6.2 - Email Classification Formatting](#exercise-62---email-classification-formatting)
"""

"""
### Exercise 6.1 - Classifying Emails
In this exercise, we'll be instructing Claude to sort emails into the following categories:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)

For the first part of the exercise, change the `PROMPT` to **make Claude output the correct classification and ONLY the classification**. Your answer needs to **include the letter (A - D) of the correct choice, with the parentheses, as well as the name of the category**.

Refer to the comments beside each email in the `EMAILS` list to know which category that email should be classified under.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "A\) P",
    "B": "B\) B",
    "C": "C\) B",
    "D": "D\) O"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_6_1_hint)

"""
Still stuck? Run the cell below for an example solution.						
"""

print(hints.exercise_6_1_solution)

"""
### Exercise 6.2 - Email Classification Formatting
In this exercise, we're going to refine the output of the above prompt to yield an answer formatted exactly how we want it. 

Use your favorite output formatting technique to make Claude wrap JUST the letter of the correct classification in `<answer></answer>` tags. For instance, the answer to the first email should contain the exact string `<answer>B</answer>`.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "<answer>A</answer>",
    "B": "<answer>B</answer>",
    "C": "<answer>C</answer>",
    "D": "<answer>D</answer>"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_6_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: AmazonBedrock/boto3/07_Using_Examples_Few-Shot_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 7: Using Examples (Few-Shot Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(prompt, system='', prefill=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages":[
              {"role": "user", "content": prompt},
              {"role": "assistant", "content": prefill}
            ],
            "temperature": 0.0,
            "top_p": 1,
            "system": system
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

**Giving Claude examples of how you want it to behave (or how you want it not to behave) is extremely effective** for:
- Getting the right answer
- Getting the answer in the right format

This sort of prompting is also called "**few shot prompting**". You might also encounter the phrase "zero-shot" or "n-shot" or "one-shot". The number of "shots" refers to how many examples are used within the prompt.
"""

"""
### Examples

Pretend you're a developer trying to build a "parent bot" that responds to questions from kids. **Claude's default response is quite formal and robotic**. This is going to break a child's heart.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

"""
You could take the time to describe your desired tone, but it's much easier just to **give Claude a few examples of ideal responses**.
"""

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

"""
In the following formatting example, we could walk Claude step by step through a set of formatting instructions on how to extract names and professions and then format them exactly the way we want, or we could just **provide Claude with some correctly-formatted examples and Claude can extrapolate from there**. Note the `<individuals>` in the `assistant` turn to start Claude off on the right foot.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 7.1 - Email Formatting via Examples](#exercise-71---email-formatting-via-examples)
"""

"""
### Exercise 7.1 - Email Formatting via Examples
We're going to redo Exercise 6.2, but this time, we're going to edit the `PROMPT` to use "few-shot" examples of emails + proper classification (and formatting) to get Claude to output the correct answer. We want the *last* letter of Claude's output to be the letter of the category.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.

Remember that these are the categories for the emails:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)								
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(ans, response[-1])) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

print(hints.exercise_7_1_hint)

"""
Still stuck? Run the cell below for an example solution.
"""

print(hints.exercise_7_1_solution)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: AmazonBedrock/boto3/10_1_Appendix_Chaining_Prompts.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.1: Chaining Prompts

- [Lesson](#lesson)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Import python's built-in regular expression library
import re
import boto3
import json

# Retrieve the MODEL_NAME variable from the IPython store
%store -r MODEL_NAME
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(messages, system_prompt=''):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": messages,
            "temperature": 0.0,
            "top_p": 1,
            "system": system_prompt
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

The saying goes, "Writing is rewriting." It turns out, **Claude can often improve the accuracy of its response when asked to do so**!

There are many ways to prompt Claude to "think again". The ways that feel natural to ask a human to double check their work will also generally work for Claude. (Check out our [prompt chaining documentation](https://docs.anthropic.com/claude/docs/chain-prompts) for further examples of when and how to use prompt chaining.)
"""

"""
### Examples

In this example, we ask Claude to come up with ten words... but one or more of them isn't a real word.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
**Asking Claude to make its answer more accurate** fixes the error! 

Below, we've pulled down Claude's incorrect response from above and added another turn to the conversation asking Claude to fix its previous answer.
"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
But is Claude revising its answer just because we told it to? What if we start off with a correct answer already? Will Claude lose its confidence? Here, we've placed a correct response in the place of `first_response` and asked it to double check again.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
You may notice that if you generate a respnse from the above block a few times, Claude leaves the words as is most of the time, but still occasionally changes the words even though they're all already correct. What can we do to mitigate this? Per Chapter 8, we can give Claude an out! Let's try this one more time.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Try generating responses from the above code a few times to see that Claude is much better at sticking to its guns now.

You can also use prompt chaining to **ask Claude to make its responses better**. Below, we asked Claude to first write a story, and then improve the story it wrote. Your personal tastes may vary, but many might agree that Claude's second version is better.

First, let's generate Claude's first version of the story.
"""

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
Now let's have Claude improve on its first draft.
"""

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
This form of substitution is very powerful. We've been using substitution placeholders to pass in lists, words, Claude's former responses, and so on. You can also **use substitution to do what we call "function calling," which is asking Claude to perform some function, and then taking the results of that function and asking Claude to do even more afterward with the results**. It works like any other substitution. More on this in the next appendix.

Below is one more example of taking the results of one call to Claude and plugging it into another, longer call. Let's start with the first prompt (which includes prefilling Claude's response this time).
"""

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

"""
Let's pass this list of names into another prompt.
"""

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Now that you've learned about prompt chaining, head over to Appendix 10.2 to learn how to implement function calling using prompt chaining.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))



================================================
FILE: AmazonBedrock/boto3/10_2_Appendix_Tool_Use.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.2: Tool Use

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

# Rewrittten to call Claude 3 Sonnet, which is generally better at tool use, and include stop_sequences
# Import python's built-in regular expression library
import re
import boto3
import json

# Import the hints module from the utils package
import os
import sys
module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import hints

# Override the MODEL_NAME variable in the IPython store to use Sonnet instead of the Haiku model
MODEL_NAME='anthropic.claude-3-sonnet-20240229-v1:0'
%store -r AWS_REGION

client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(messages, system_prompt="", prefill="", stop_sequences=None):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "temperature": 0.0,
            "top_p": 1,
            "messages":messages,
            "system": system_prompt,
            "stop_sequences": stop_sequences
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

"""
---

## Lesson

While it might seem conceptually complex at first, tool use, a.k.a. function calling, is actually quite simple! You already know all the skills necessary to implement tool use, which is really just a combination of substitution and prompt chaining.

In previous substitution exercises, we substituted text into prompts. With tool use, we substitute tool or function results into prompts. Claude can't literally call or access tools and functions. Instead, we have Claude:
1. Output the tool name and arguments it wants to call
2. Halt any further response generation while the tool is called
3. Then we reprompt with the appended tool results
"""

"""
Function calling is useful because it expands Claude's capabilities and enables Claude to handle much more complex, multi-step tasks.
Some examples of functions you can give Claude:
- Calculator
- Word counter
- SQL database querying and data retrieval
- Weather API
"""

"""
You can get Claude to do tool use by combining these two elements:

1. A system prompt, in which we give Claude an explanation of the concept of tool use as well as a detailed descriptive list of the tools it has access to
2. The control logic with which to orchestrate and execute Claude's tool use requests
"""

"""
### Tool use roadmap

*This lesson teaches our current tool use format. However, we will be updating and improving tool use functionality in the near future, including:*
* *A more streamlined format for function definitions and calls*
* *More robust error handilgj and edge case coverage*
* *Tighter integration with the rest of our API*
* *Better reliability and performance, especially for more complex tool use tasks*
"""

"""
### Examples

To enable tool use in Claude, we start with the system prompt. In this special tool use system prompt, wet tell Claude:
* The basic premise of tool use and what it entails
* How Claude can call and use the tools it's been given
* A detailed list of tools it has access to in this specific scenario 

Here's the first part of the system prompt, explaining tool use to Claude. This part of the system prompt is generalizable across all instances of prompting Claude for tool use. The tool calling structure we're giving Claude (`<function_calls> [...] </function_calls>`) is a structure Claude has been specifically trained to use, so we recommend that you stick with this.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

"""
Here's the second part of the system prompt, which defines the exact tools Claude has access to in this specific situation. In this example, we will be giving Claude a calculator tool, which takes three parameters: two operands and an operator. 

Then we combine the two parts of the system prompt.
"""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

"""
Now we can give Claude a question that requires use of the `calculator` tool. We will use `<function_calls\>` in `stop_sequences` to detect if and when Claude calls the function.
"""

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Now, we can extract out the parameters from Claude's function call and actually run the function on Claude's behalf.

First we'll define the function's code.
"""

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

"""
Then we'll extract the parameters from Claude's function call response. If all the parameters exist, we run the calculator tool.
"""

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

"""
Now that we have a result, we have to properly format that result so that when we pass it back to Claude, Claude understands what tool that result is in relation to. There is a set format for this that Claude has been trained to recognize:
```
<function_results>
<result>
<tool_name>{TOOL_NAME}</tool_name>
<stdout>
{TOOL_RESULT}
</stdout>
</result>
</function_results>
```

Run the cell below to format the above tool result into this structure.
"""

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

"""
Now all we have to do is send this result back to Claude by appending the result to the same message chain as before, and we're good!
"""

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

"""
Congratulations on running an entire tool use chain end to end!

Now what if we give Claude a question that doesn't that doesn't require using the given tool at all?
"""

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Success! As you can see, Claude knew not to call the function when it wasn't needed.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 10.2.1 - SQL](#exercise-1021---SQL)
"""

"""
### Exercise 10.2.1 - SQL
In this exercise, you'll be writing a tool use prompt for querying and writing to the world's smallest "database". Here's the initialized database, which is really just a dictionary.
"""

db = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "products": [
        {"id": 1, "name": "Widget", "price": 9.99},
        {"id": 2, "name": "Gadget", "price": 14.99},
        {"id": 3, "name": "Doohickey", "price": 19.99}
    ]
}

"""
And here is the code for the functions that write to and from the database.
"""

def get_user(user_id):
    for user in db["users"]:
        if user["id"] == user_id:
            return user
    return None

def get_product(product_id):
    for product in db["products"]:
        if product["id"] == product_id:
            return product
    return None

def add_user(name, email):
    user_id = len(db["users"]) + 1
    user = {"id": user_id, "name": name, "email": email}
    db["users"].append(user)
    return user

def add_product(name, price):
    product_id = len(db["products"]) + 1
    product = {"id": product_id, "name": name, "price": price}
    db["products"].append(product)
    return product

"""
To solve the exercise, start by defining a system prompt like `system_prompt_tools_specific_tools` above. Make sure to include the name and description of each tool, along with the name and type and description of each parameter for each function. We've given you some starting scaffolding below.
"""

system_prompt_tools_specific_tools_sql = """
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools_sql

"""
When you're ready, you can try out your tool definition system prompt on the examples below. Just run the below cell!
"""

examples = [
    "Add a user to the database named Deborah.",
    "Add a product to the database named Thingo",
    "Tell me the name of User 2",
    "Tell me the name of Product 3"
]

for example in examples:
    message = {
        "role": "user",
        "content": example
    }

    # Get & print Claude's response
    function_calling_response = get_completion([message], system_prompt=system_prompt, stop_sequences=stop_sequences)
    print(example, "\n----------\n\n", function_calling_response, "\n*********\n*********\n*********\n\n")

"""
If you did it right, the function calling messages should call the `add_user`, `add_product`, `get_user`, and `get_product` functions correctly.

For extra credit, add some code cells and write parameter-parsing code. Then call the functions with the parameters Claude gives you to see the state of the "database" after the call.
"""

"""
❓ If you want to see a possible solution, run the cell below!
"""

print(hints.exercise_10_2_1_solution)

"""
### Congrats!

Congratulations on learning tool use and function calling! Head over to the last appendix section if you would like to learn more about search & RAG.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)



================================================
FILE: AmazonBedrock/boto3/10_3_Appendix_Empirical_Performance_Eval.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Evaluating AI Models: Code, Human, and Model-Based Grading

In this notebook, we'll delve into a trio of widely-used techniques for assessing the effectiveness of AI models, like Claude v3:

1. Code-based grading
2. Human grading
3. Model-based grading

We'll illustrate each approach through examples and examine their respective advantages and limitations, when gauging AI performance.
"""

"""
## Code-Based Grading Example: Sentiment Analysis

In this example, we'll evaluate Claude's ability to classify the sentiment of movie reviews as positive or negative. We can use code to check if the model's output matches the expected sentiment.
"""

# Import python's built-in regular expression library
import re

# Import boto3 and json
import boto3
import json

# Store the model name and AWS region for later use
MODEL_NAME = "anthropic.claude-3-haiku-20240307-v1:0"
AWS_REGION = "us-west-2"

%store MODEL_NAME
%store AWS_REGION

# Function to build the input prompt for sentiment analysis
def build_input_prompt(review):
    user_content = f"""Classify the sentiment of the following movie review as either 'positive' or 'negative' provide only one of those two choices:
    <review>{review}</review>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "review": "This movie was amazing! The acting was superb and the plot kept me engaged from start to finish.",
        "golden_answer": "positive"
    },
    {
        "review": "I was thoroughly disappointed by this film. The pacing was slow and the characters were one-dimensional.",
        "golden_answer": "negative"
    }
]

# Function to get completions from the model
client = boto3.client('bedrock-runtime',region_name=AWS_REGION)

def get_completion(messages):
    body = json.dumps(
        {
            "anthropic_version": '',
            "max_tokens": 2000,
            "messages": messages,
            "temperature": 0.5,
            "top_p": 1,
        }
    )
    response = client.invoke_model(body=body, modelId=MODEL_NAME)
    response_body = json.loads(response.get('body').read())

    return response_body.get('content')[0].get('text')

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["review"])) for item in eval]

# Print the outputs and golden answers
for output, question in zip(outputs, eval):
    print(f"Review: {question['review']}\nGolden Answer: {question['golden_answer']}\nOutput: {output}\n")

# Function to grade the completions
def grade_completion(output, golden_answer):
    return output.lower() == golden_answer.lower()

# Grade the completions and print the accuracy
grades = [grade_completion(output, item["golden_answer"]) for output, item in zip(outputs, eval)]
print(f"Accuracy: {sum(grades) / len(grades) * 100}%")

"""
## Human Grading Example: Essay Scoring

Some tasks, like scoring essays, are difficult to evaluate with code alone. In this case, we can provide guidelines for human graders to assess the model's output.
"""

# Function to build the input prompt for essay generation
def build_input_prompt(topic):
    user_content = f"""Write a short essay discussing the following topic:
    <topic>{topic}</topic>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "topic": "The importance of education in personal development and societal progress",
        "golden_answer": "A high-scoring essay should have a clear thesis, well-structured paragraphs, and persuasive examples discussing how education contributes to individual growth and broader societal advancement."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["topic"])) for item in eval]

# Print the outputs and golden answers
for output, item in zip(outputs, eval):
    print(f"Topic: {item['topic']}\n\nGrading Rubric:\n {item['golden_answer']}\n\nModel Output:\n{output}\n")

"""
## Model-Based Grading Examples

We can use Claude to grade its own outputs by providing the model's response and a grading rubric. This allows us to automate the evaluation of tasks that would typically require human judgment.
"""

"""
### Example 1: Summarization

In this example, we'll use Claude to assess the quality of a summary it generated. This can be useful when you need to evaluate the model's ability to capture key information from a longer text concisely and accurately. By providing a rubric that outlines the essential points that should be covered, we can automate the grading process and quickly assess the model's performance on summarization tasks.
"""

# Function to build the input prompt for summarization
def build_input_prompt(text):
    user_content = f"""Please summarize the main points of the following text:
    <text>{text}</text>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing summary quality
def build_grader_prompt(output, rubric):
    user_content = f"""Assess the quality of the following summary based on this rubric:
    <rubric>{rubric}</rubric>
    <summary>{output}</summary>
    Provide a score from 1-5, where 1 is poor and 5 is excellent."""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "text": "The Magna Carta, signed in 1215, was a pivotal document in English history. It limited the powers of the monarchy and established the principle that everyone, including the king, was subject to the law. This laid the foundation for constitutional governance and the rule of law in England and influenced legal systems worldwide.",
        "golden_answer": "A high-quality summary should concisely capture the key points: 1) The Magna Carta's significance in English history, 2) Its role in limiting monarchical power, 3) Establishing the principle of rule of law, and 4) Its influence on legal systems around the world."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["text"])) for item in eval]

# Grade the completions
grades = [get_completion(build_grader_prompt(output, item["golden_answer"])) for output, item in zip(outputs, eval)]

# Print the summary quality score
print(f"Summary quality score: {grades[0]}")

"""
### Example 2: Fact-Checking

In this example, we'll use Claude to fact-check a claim and then assess the accuracy of its fact-checking. This can be useful when you need to evaluate the model's ability to distinguish between accurate and inaccurate information. By providing a rubric that outlines the key points that should be covered in a correct fact-check, we can automate the grading process and quickly assess the model's performance on fact-checking tasks.
"""

# Function to build the input prompt for fact-checking
def build_input_prompt(claim):
    user_content = f"""Determine if the following claim is true or false:
    <claim>{claim}</claim>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing fact-check accuracy
def build_grader_prompt(output, rubric):
    user_content = f"""Based on the following rubric, assess whether the fact-check is correct:
    <rubric>{rubric}</rubric>
    <fact-check>{output}</fact-check>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "claim": "The Great Wall of China is visible from space.",
        "golden_answer": "A correct fact-check should state that this claim is false. While the Great Wall is an impressive structure, it is not visible from space with the naked eye."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["claim"])) for item in eval]

grades = []
for output, item in zip(outputs, eval):
    # Print the claim, fact-check, and rubric
    print(f"Claim: {item['claim']}\n")
    print(f"Fact-check: {output}]\n")
    print(f"Rubric: {item['golden_answer']}\n")
    
    # Grade the fact-check
    grader_prompt = build_grader_prompt(output, item["golden_answer"])
    grade = get_completion(grader_prompt)
    grades.append("correct" in grade.lower())

# Print the fact-checking accuracy
accuracy = sum(grades) / len(grades)
print(f"Fact-checking accuracy: {accuracy * 100}%")

"""
### Example 3: Tone Analysis

In this example, we'll use Claude to analyze the tone of a given text and then assess the accuracy of its analysis. This can be useful when you need to evaluate the model's ability to identify and interpret the emotional content and attitudes expressed in a piece of text. By providing a rubric that outlines the key aspects of tone that should be identified, we can automate the grading process and quickly assess the model's performance on tone analysis tasks.
"""

# Function to build the input prompt for tone analysis
def build_input_prompt(text):
    user_content = f"""Analyze the tone of the following text:
    <text>{text}</text>"""
    return [{"role": "user", "content": user_content}]

# Function to build the grader prompt for assessing tone analysis accuracy
def build_grader_prompt(output, rubric):
    user_content = f"""Assess the accuracy of the following tone analysis based on this rubric:
    <rubric>{rubric}</rubric>
    <tone-analysis>{output}</tone-analysis>"""
    return [{"role": "user", "content": user_content}]

# Define the evaluation data
eval = [
    {
        "text": "I can't believe they canceled the event at the last minute. This is completely unacceptable and unprofessional!",
        "golden_answer": "The tone analysis should identify the text as expressing frustration, anger, and disappointment. Key words like 'can't believe', 'last minute', 'unacceptable', and 'unprofessional' indicate strong negative emotions."
    }
]

# Get completions for each input
outputs = [get_completion(build_input_prompt(item["text"])) for item in eval]

# Grade the completions
grades = [get_completion(build_grader_prompt(output, item["golden_answer"])) for output, item in zip(outputs, eval)]

# Print the tone analysis quality
print(f"Tone analysis quality: {grades[0]}")

"""
These examples demonstrate how code-based, human, and model-based grading can be used to evaluate AI models like Claude on various tasks. The choice of evaluation method depends on the nature of the task and the resources available. Model-based grading offers a promising approach for automating the assessment of complex tasks that would otherwise require human judgment.
"""



================================================
FILE: AmazonBedrock/boto3/10_4_Appendix_Search_and_Retrieval.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.4: Search & Retrieval

Did you know you can use Claude to **search through Wikipedia for you**? Claude can find and retrieve articles, at which point you can also use Claude to summarize and synthesize them, write novel content from what it found, and much more. And not just Wikipedia! You can also search over your own docs, whether stored as plain text or embedded in a vector datastore.

See our [RAG cookbook examples](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/Wikipedia/wikipedia-search-cookbook.ipynb) to learn how to supplement Claude's knowledge and improve the accuracy and relevance of Claude's responses with data retrieved from vector databases, Wikipedia, the internet, and more. There, you can also learn about how to use certain [embeddings](https://docs.anthropic.com/claude/docs/embeddings) and vector database tools.

If you are interested in learning about advanced RAG architectures using Claude, check out our [Claude 3 technical presentation slides on RAG architectures](https://docs.google.com/presentation/d/1zxkSI7lLUBrZycA-_znwqu8DDyVhHLkQGScvzaZrUns/edit#slide=id.g2c736259dac_63_782).
"""



================================================
FILE: AmazonBedrock/cloudformation/workshop-v1-final-cfn.yml
================================================
AWSTemplateFormatVersion: '2010-09-09'
Description: 'CloudFormation template to create a Jupyter notebook in SageMaker with an execution role and Anthropic Prompt Eng. Repo'

Parameters:
  NotebookName:
    Type: String
    Default: 'PromptEngWithAnthropicNotebook'
  DefaultRepoUrl:
    Type: String
    Default: 'https://github.com/aws-samples/prompt-engineering-with-anthropic-claude-v-3.git'

Resources:
  SageMakerExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service:
                - sagemaker.amazonaws.com
            Action:
              - sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess

  KmsKey:
    Type: AWS::KMS::Key
    Properties:
      Description: 'KMS key for SageMaker notebook'
      KeyPolicy:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              AWS: !Sub 'arn:aws:iam::${AWS::AccountId}:root'
            Action: 'kms:*'
            Resource: '*'
      EnableKeyRotation: true

  KmsKeyAlias:
    Type: AWS::KMS::Alias
    Properties:
      AliasName: !Sub 'alias/${NotebookName}-kms-key'
      TargetKeyId: !Ref KmsKey

  SageMakerNotebookInstance:
    Type: AWS::SageMaker::NotebookInstance
    Properties:
      InstanceType: ml.t3.large
      NotebookInstanceName: !Ref NotebookName
      RoleArn: !GetAtt SageMakerExecutionRole.Arn
      DefaultCodeRepository: !Ref DefaultRepoUrl
      KmsKeyId: !GetAtt KmsKey.Arn

Outputs:
  NotebookInstanceName:
    Description: The name of the created SageMaker Notebook Instance
    Value: !Ref SageMakerNotebookInstance
  ExecutionRoleArn:
    Description: The ARN of the created SageMaker Execution Role
    Value: !GetAtt SageMakerExecutionRole.Arn
  KmsKeyArn:
    Description: The ARN of the created KMS Key for the notebook
    Value: !GetAtt KmsKey.Arn



================================================
FILE: AmazonBedrock/utils/__init__.py
================================================
[Empty file]


================================================
FILE: AmazonBedrock/utils/hints.py
================================================
exercise_1_1_hint = """The grading function in this exercise is looking for an answer that contains the exact Arabic numerals "1", "2", and "3".
You can often get Claude to do what you want simply by asking."""

exercise_1_2_hint = """The grading function in this exercise is looking for answers that contain "soo" or "giggles".
There are many ways to solve this, just by asking!"""

exercise_2_1_hint ="""The grading function in this exercise is looking for any answer that includes the word "hola".
Ask Claude to reply in Spanish like you would when speaking with a human. It's that simple!"""

exercise_2_2_hint = """The grading function in this exercise is looking for EXACTLY "Michael Jordan".
How would you ask another human to do this? Reply with no other words? Reply with only the name and nothing else? There are several ways to approach this answer."""

exercise_2_3_hint = """The grading function in this cell is looking for a response that is equal to or greater than 800 words.
Because LLMs aren't great at counting words yet, you may have to overshoot your target."""

exercise_3_1_hint = """The grading function in this exercise is looking for an answer that includes the words "incorrect" or "not correct".
Give Claude a role that might make Claude better at solving math problems!"""

exercise_4_1_hint = """The grading function in this exercise is looking for a solution that includes the words "haiku" and "pig".
Don't forget to include the exact phrase "{TOPIC}" wherever you want the topic to be substituted in. Changing the "TOPIC" variable value should make Claude write a haiku about a different topic."""

exercise_4_2_hint = """The grading function in this exercise is looking for a response that includes the word "brown".
If you surround "{QUESTION}" in XML tags, how does that change Claude's response?"""

exercise_4_3_hint = """The grading function in this exercise is looking for a response that includes the word "brown".
Try removing one word or section of characters at a time, starting with the parts that make the least sense. Doing this one word at a time will also help you see just how much Claude can or can't parse and understand."""

exercise_5_1_hint = """The grading function for this exercise is looking for a response that includes the word "Warrior".
Write more words in Claude's voice to steer Claude to act the way you want it to. For instance, instead of "Stephen Curry is the best because," you could write "Stephen Curry is the best and here are three reasons why. 1:"""

exercise_5_2_hint = """The grading function looks for a response of over 5 lines in length that includes the words "cat" and "<haiku>".
Start simple. Currently, the prompt asks Claude for one haiku. You can change that and ask for two (or even more). Then if you run into formatting issues, change your prompt to fix that after you've already gotten Claude to write more than one haiku."""

exercise_5_3_hint = """The grading function in this exercise is looking for a response that contains the words "tail", "cat", and "<haiku>".
It's helpful to break this exercise down to several steps.								
1.	Modify the initial prompt template so that Claude writes two poems.							
2.	Give Claude indicators as to what the poems will be about, but instead of writing in the subjects directly (e.g., dog, cat, etc.), replace those subjects with the keywords "{ANIMAL1}" and "{ANIMAL2}".							
3.	Run the prompt and make sure that the full prompt with variable substitutions has all the words correctly substituted. If not, check to make sure your {bracket} tags are spelled correctly and formatted correctly with single moustache brackets."""

exercise_6_1_hint = """The grading function in this exercise is looking for the correct categorization letter + the closing parentheses and the first letter of the name of the category, such as "C) B" or "B) B" etc.
Let's take this exercise step by step:										
1.	How will Claude know what categories you want to use? Tell it! Include the four categories you want directly in the prompt. Be sure to include the parenthetical letters as well for easy classification. Feel free to use XML tags to organize your prompt and make clear to Claude where the categories begin and end.									
2.	Try to cut down on superfluous text so that Claude immediately answers with the classification and ONLY the classification. There are several ways to do this, from speaking for Claude (providing anything from the beginning of the sentence to a single open parenthesis so that Claude knows you want the parenthetical letter as the first part of the answer) to telling Claude that you want the classification and only the classification, skipping the preamble.
Refer to Chapters 2 and 5 if you want a refresher on these techniques.							
3.	Claude may still be incorrectly categorizing or not including the names of the categories when it answers. Fix this by telling Claude to include the full category name in its answer.)								
4.	Be sure that you still have {email} somewhere in your prompt template so that we can properly substitute in emails for Claude to evaluate."""

exercise_6_1_solution = """
USER TURN
Please classify this email into the following categories: {email}

Do not include any extra words except the category.

<categories>
(A) Pre-sale question
(B) Broken or defective item
(C) Billing question
(D) Other (please explain)
</categories>

ASSISTANT TURN
(
"""

exercise_6_2_hint = """The grading function in this exercise is looking for only the correct letter wrapped in <answer> tags, such as "<answer>B</answer>". The correct categorization letters are the same as in the above exercise.
Sometimes the simplest way to go about this is to give Claude an example of how you want its output to look. Just don't forget to wrap your example in <example></example> tags! And don't forget that if you prefill Claude's response with anything, Claude won't actually output that as part of its response."""

exercise_7_1_hint = """You're going to have to write some example emails and classify them for Claude (with the exact formatting you want). There are multiple ways to do this. Here are some guidelines below.										
1.	Try to have at least two example emails. Claude doesn't need an example for all categories, and the examples don't have to be long. It's more helpful to have examples for whatever you think the trickier categories are (which you were asked to think about at the bottom of Chapter 6 Exercise 1). XML tags will help you separate out your examples from the rest of your prompt, although it's unnecessary.									
2.	Make sure your example answer formatting is exactly the format you want Claude to use, so Claude can emulate the format as well. This format should make it so that Claude's answer ends in the letter of the category. Wherever you put the {email} placeholder, make sure that it's formatted exactly like your example emails.									
3.	Make sure you still have the categories listed within the prompt itself, otherwise Claude won't know what categories to reference, as well as {email} as a placeholder for substitution."""

exercise_7_1_solution = """
USER TURN
Please classify emails into the following categories, and do not include explanations: 
<categories>
(A) Pre-sale question
(B) Broken or defective item
(C) Billing question
(D) Other (please explain)
</categories>

Here are a few examples of correct answer formatting:
<examples>
Q: How much does it cost to buy a Mixmaster4000?
A: The correct category is: A

Q: My Mixmaster won't turn on.
A: The correct category is: B

Q: Please remove me from your mailing list.
A: The correct category is: D
</examples>

Here is the email for you to categorize: {email}

ASSISTANT TURN
The correct category is:
"""
exercise_8_1_hint = """The grading function in this exercise is looking for a response that contains the phrase "I do not", "I don't", or "Unfortunately".
What should Claude do if it doesn't know the answer?"""

exercise_8_2_hint = """The grading function in this exercise is looking for a response that contains the phrase "49-fold".
Make Claude show its work and thought process first by extracting relevant quotes and seeing whether or not the quotes provide sufficient evidence. Refer back to the Chapter 8 Lesson if you want a refresher."""

exercise_9_1_solution = """
You are a master tax acountant. Your task is to answer user questions using any provided reference documentation.

Here is the material you should use to answer the user's question:
<docs>
{TAX_CODE}
</docs>

Here is an example of how to respond:
<example>
<question>
What defines a "qualified" employee?
</question>
<answer>
<quotes>For purposes of this subsection—
(A)In general
The term "qualified employee" means any individual who—
(i)is not an excluded employee, and
(ii)agrees in the election made under this subsection to meet such requirements as are determined by the Secretary to be necessary to ensure that the withholding requirements of the corporation under chapter 24 with respect to the qualified stock are met.</quotes>

<answer>According to the provided documentation, a "qualified employee" is defined as an individual who:

1. Is not an "excluded employee" as defined in the documentation.
2. Agrees to meet the requirements determined by the Secretary to ensure the corporation's withholding requirements under Chapter 24 are met with respect to the qualified stock.</answer>
</example>

First, gather quotes in <quotes></quotes> tags that are relevant to answering the user's question. If there are no quotes, write "no relevant quotes found".

Then insert two paragraph breaks before answering the user question within <answer></answer> tags. Only answer the user's question if you are confident that the quotes in <quotes></quotes> tags support your answer. If not, tell the user that you unfortunately do not have enough information to answer the user's question.

Here is the user question: {QUESTION}
"""

exercise_9_2_solution = """
You are Codebot, a helpful AI assistant who finds issues with code and suggests possible improvements.

Act as a Socratic tutor who helps the user learn.

You will be given some code from a user. Please do the following:
1. Identify any issues in the code. Put each issue inside separate <issues> tags.
2. Invite the user to write a revised version of the code to fix the issue.

Here's an example:

<example>
<code>
def calculate_circle_area(radius):
    return (3.14 * radius) ** 2
</code>
<issues>
<issue>
3.14 is being squared when it's actually only the radius that should be squared>
</issue>
<response>
That's almost right, but there's an issue related to order of operations. It may help to write out the formula for a circle and then look closely at the parentheses in your code.
</response>
</example>

Here is the code you are to analyze:

<code>
{CODE}
</code>

Find the relevant issues and write the Socratic tutor-style response. Do not give the user too much help! Instead, just give them guidance so they can find the correct solution themselves.

Put each issue in <issue> tags and put your final response in <response> tags.
"""

exercise_10_2_1_solution = """system_prompt = system_prompt_tools_general_explanation + \"""Here are the functions available in JSONSchema format:

<tools>

<tool_description>
<tool_name>get_user</tool_name>
<description>
Retrieves a user from the database by their user ID.
</description>
<parameters>
<parameter>
<name>user_id</name>
<type>int</type>
<description>The ID of the user to retrieve.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>get_product</tool_name>
<description>
Retrieves a product from the database by its product ID.
</description>
<parameters>
<parameter>
<name>product_id</name>
<type>int</type>
<description>The ID of the product to retrieve.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>add_user</tool_name>
<description>
Adds a new user to the database.
</description>
<parameters>
<parameter>
<name>name</name>
<type>str</type>
<description>The name of the user.</description>
</parameter>
<parameter>
<name>email</name>
<type>str</type>
<description>The email address of the user.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>add_product</tool_name>
<description>
Adds a new product to the database.
</description>
<parameters>
<parameter>
<name>name</name>
<type>str</type>
<description>The name of the product.</description>
</parameter>
<parameter>
<name>price</name>
<type>float</type>
<description>The price of the product.</description>
</parameter>
</parameters>
</tool_description>

</tools>
"""


================================================
FILE: Anthropic 1P/00_Tutorial_How-To.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Tutorial How-To

This tutorial **requires an API key** for interaction. If you don't have an API key, you can sign up for one via the [Anthropic Console](https://console.anthropic.com/) or view our [static tutorial answer key](https://docs.google.com/spreadsheets/u/0/d/1jIxjzUWG-6xBVIa2ay6yDpLyeuOh_hR_ZB75a47KX_E/edit) instead.
"""

"""
## How to get started

1. Clone this repository to your local machine.

2. Install the required dependencies by running the following command:
 
"""

!pip install anthropic

"""
3. Set up your API key and model name. Replace `"your_api_key_here"` with your actual Anthropic API key.
"""

API_KEY = "your_api_key_here"
MODEL_NAME = "claude-3-haiku-20240307"

# Stores the API_KEY & MODEL_NAME variables for use across notebooks within the IPython store
%store API_KEY
%store MODEL_NAME

"""
4. Run the notebook cells in order, following the instructions provided.
"""

"""
---

## Usage Notes & Tips 💡

- This course uses Claude 3 Haiku with temperature 0. We will talk more about temperature later in the course. For now, it's enough to understand that these settings yield more deterministic results. All prompt engineering techniques in this course also apply to previous generation legacy Claude models such as Claude 2 and Claude Instant 1.2.

- You can use `Shift + Enter` to execute the cell and move to the next one.

- When you reach the bottom of a tutorial page, navigate to the next numbered file in the folder, or to the next numbered folder if you're finished with the content within that chapter file.

### The Anthropic SDK & the Messages API
We will be using the [Anthropic python SDK](https://docs.anthropic.com/claude/reference/client-sdks) and the [Messages API](https://docs.anthropic.com/claude/reference/messages_post) throughout this tutorial. 

Below is an example of what running a prompt will look like in this tutorial. First, we create `get_completion`, which is a helper function that sends a prompt to Claude and returns Claude's generated response. Run that cell now.
"""

import anthropic

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

"""
Now we will write out an example prompt for Claude and print Claude's output by running our `get_completion` helper function. Running the cell below will print out a response from Claude beneath it.

Feel free to play around with the prompt string to elicit different responses from Claude.
"""

# Prompt
prompt = "Hello, Claude!"

# Get Claude's response
print(get_completion(prompt))

"""
The `API_KEY` and `MODEL_NAME` variables defined earlier will be used throughout the tutorial. Just make sure to run the cells for each tutorial page from top to bottom.
"""



================================================
FILE: Anthropic 1P/01_Basic_Prompt_Structure.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 1: Basic Prompt Structure

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str, system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

Anthropic offers two APIs, the legacy [Text Completions API](https://docs.anthropic.com/claude/reference/complete_post) and the current [Messages API](https://docs.anthropic.com/claude/reference/messages_post). For this tutorial, we will be exclusively using the Messages API.

At minimum, a call to Claude using the Messages API requires the following parameters:
- `model`: the [API model name](https://docs.anthropic.com/claude/docs/models-overview#model-recommendations) of the model that you intend to call

- `max_tokens`: the maximum number of tokens to generate before stopping. Note that Claude may stop before reaching this maximum. This parameter only specifies the absolute maximum number of tokens to generate. Furthermore, this is a *hard* stop, meaning that it may cause Claude to stop generating mid-word or mid-sentence.

- `messages`: an array of input messages. Our models are trained to operate on alternating `user` and `assistant` conversational turns. When creating a new `Message`, you specify the prior conversational turns with the messages parameter, and the model then generates the next `Message` in the conversation.
  - Each input message must be an object with a `role` and `content`. You can specify a single `user`-role message, or you can include multiple `user` and `assistant` messages (they must alternate, if so). The first message must always use the user `role`.

There are also optional parameters, such as:
- `system`: the system prompt - more on this below.
  
- `temperature`: the degree of variability in Claude's response. For these lessons and exercises, we have set `temperature` to 0.

For a complete list of all API parameters, visit our [API documentation](https://docs.anthropic.com/claude/reference/messages_post).
"""

"""
### Examples

Let's take a look at how Claude responds to some correctly-formatted prompts. For each of the following cells, run the cell (`shift+enter`), and Claude's response will appear below the block.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now let's take a look at some prompts that do not include the correct Messages API formatting. For these malformatted prompts, the Messages API returns an error.

First, we have an example of a Messages API call that lacks `role` and `content` fields in the `messages` array.
"""

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"Hi Claude, how are you?"}
        ]
    )

# Print Claude's response
print(response[0].text)

"""
Here's a prompt that fails to alternate between the `user` and `assistant` roles.
"""

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ]
    )

# Print Claude's response
print(response[0].text)

"""
`user` and `assistant` messages **MUST alternate**, and messages **MUST start with a `user` turn**. You can have multiple `user` & `assistant` pairs in a prompt (as if simulating a multi-turn conversation). You can also put words into a terminal `assistant` message for Claude to continue from where you left off (more on that in later chapters).

#### System Prompts

You can also use **system prompts**. A system prompt is a way to **provide context, instructions, and guidelines to Claude** before presenting it with a question or task in the "User" turn. 

Structurally, system prompts exist separately from the list of `user` & `assistant` messages, and thus belong in a separate `system` parameter (take a look at the structure of the `get_completion` helper function in the [Setup](#setup) section of the notebook). 

Within this tutorial, wherever we might utilize a system prompt, we have provided you a `system` field in your completions function. Should you not want to use a system prompt, simply set the `SYSTEM_PROMPT` variable to an empty string.
"""

"""
#### System Prompt Example
"""

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
Why use a system prompt? A **well-written system prompt can improve Claude's performance** in a variety of ways, such as increasing Claude's ability to follow rules and instructions. For more information, visit our documentation on [how to use system prompts](https://docs.anthropic.com/claude/docs/how-to-use-system-prompts) with Claude.

Now we'll dive into some exercises. If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 1.1 - Counting to Three](#exercise-11---counting-to-three)
- [Exercise 1.2 - System Prompt](#exercise-12---system-prompt)
"""

"""
### Exercise 1.1 - Counting to Three
Using proper `user` / `assistant` formatting, edit the `PROMPT` below to get Claude to **count to three.** The output will also indicate whether your solution is correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    pattern = re.compile(r'^(?=.*1)(?=.*2)(?=.*3).*$', re.DOTALL)
    return bool(pattern.match(text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_1_1_hint; print(exercise_1_1_hint)

"""
### Exercise 1.2 - System Prompt

Modify the `SYSTEM_PROMPT` to make Claude respond like it's a 3 year old child.
"""

# System prompt - this is the only field you should change
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "How big is the sky?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search(r"giggles", text) or re.search(r"soo", text))

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_1_2_hint; print(exercise_1_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Hi Claude, how are you?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Can you tell me the color of the ocean?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "What year was Celine Dion born in?"

# Print Claude's response
print(get_completion(PROMPT))

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"Hi Claude, how are you?"}
        ]
    )

# Print Claude's response
print(response[0].text)

# Get Claude's response
response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        messages=[
          {"role": "user", "content": "What year was Celine Dion born in?"},
          {"role": "user", "content": "Also, can you tell me some other facts about her?"}
        ]
    )

# Print Claude's response
print(response[0].text)

# System prompt
SYSTEM_PROMPT = "Your answer should always be a series of critical thinking questions that further the conversation (do not provide answers to your questions). Do not actually answer the user question."

# Prompt
PROMPT = "Why is the sky blue?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: Anthropic 1P/02_Being_Clear_and_Direct.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 2: Being Clear and Direct

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

# Note that we changed max_tokens to 4K just for this lesson to allow for longer completions in the exercises
def get_completion(prompt: str, system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=4000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

**Claude responds best to clear and direct instructions.**

Think of Claude like any other human that is new to the job. **Claude has no context** on what to do aside from what you literally tell it. Just as when you instruct a human for the first time on a task, the more you explain exactly what you want in a straightforward manner to Claude, the better and more accurate Claude's response will be."				
				
When in doubt, follow the **Golden Rule of Clear Prompting**:
- Show your prompt to a colleague or friend and have them follow the instructions themselves to see if they can produce the result you want. If they're confused, Claude's confused.				
"""

"""
### Examples

Let's take a task like writing poetry. (Ignore any syllable mismatch - LLMs aren't great at counting syllables yet.)
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

"""
This haiku is nice enough, but users may want Claude to go directly into the poem without the "Here is a haiku" preamble.

How do we achieve that? We **ask for it**!
"""

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

"""
Here's another example. Let's ask Claude who's the best basketball player of all time. You can see below that while Claude lists a few names, **it doesn't respond with a definitive "best"**.
"""

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Can we get Claude to make up its mind and decide on a best player? Yes! Just ask!
"""

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 2.1 - Spanish](#exercise-21---spanish)
- [Exercise 2.2 - One Player Only](#exercise-22---one-player-only)
- [Exercise 2.3 - Write a Story](#exercise-23---write-a-story)
"""

"""
### Exercise 2.1 - Spanish
Modify the `SYSTEM_PROMPT` to make Claude output its answer in Spanish.
"""

# System prompt - this is the only field you should chnage
SYSTEM_PROMPT = "[Replace this text]"

# Prompt
PROMPT = "Hello Claude, how are you?"

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return "hola" in text.lower()

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_2_1_hint; print(exercise_2_1_hint)

"""
### Exercise 2.2 - One Player Only

Modify the `PROMPT` so that Claude doesn't equivocate at all and responds with **ONLY** the name of one specific player, with **no other words or punctuation**. 
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return text == "Michael Jordan"

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_2_2_hint; print(exercise_2_2_hint)

"""
### Exercise 2.3 - Write a Story

Modify the `PROMPT` so that Claude responds with as long a response as you can muster. If your answer is **over 800 words**, Claude's response will be graded as correct.
"""

# Prompt - this is the only field you should change
PROMPT = "[Replace this text]"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    trimmed = text.strip()
    words = len(trimmed.split())
    return words >= 800

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_2_3_hint; print(exercise_2_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Write a haiku about robots."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Write a haiku about robots. Skip the preamble; go straight into the poem."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Who is the best basketball player of all time? Yes, there are differing opinions, but if you absolutely had to pick one player, who would it be?"

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: Anthropic 1P/03_Assigning_Roles_Role_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 3: Assigning Roles (Role Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str, system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

Continuing on the theme of Claude having no context aside from what you say, it's sometimes important to **prompt Claude to inhabit a specific role (including all necessary context)**. This is also known as role prompting. The more detail to the role context, the better.

**Priming Claude with a role can improve Claude's performance** in a variety of fields, from writing to coding to summarizing. It's like how humans can sometimes be helped when told to "think like a ______". Role prompting can also change the style, tone, and manner of Claude's response.

**Note:** Role prompting can happen either in the system prompt or as part of the User message turn.
"""

"""
### Examples

In the example below, we see that without role prompting, Claude provides a **straightforward and non-stylized answer** when asked to give a single sentence perspective on skateboarding.

However, when we prime Claude to inhabit the role of a cat, Claude's perspective changes, and thus **Claude's response tone, style, content adapts to the new role**. 

**Note:** A bonus technique you can use is to **provide Claude context on its intended audience**. Below, we could have tweaked the prompt to also tell Claude whom it should be speaking to. "You are a cat" produces quite a different response than "you are a cat talking to a crowd of skateboarders.

Here is the prompt without role prompting in the system prompt:
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Here is the same user question, except with role prompting.
"""

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
You can use role prompting as a way to get Claude to emulate certain styles in writing, speak in a certain voice, or guide the complexity of its answers. **Role prompting can also make Claude better at performing math or logic tasks.**

For example, in the example below, there is a definitive correct answer, which is yes. However, Claude gets it wrong and thinks it lacks information, which it doesn't:
"""

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

"""
Now, what if we **prime Claude to act as a logic bot**? How will that change Claude's answer? 

It turns out that with this new role assignment, Claude gets it right. (Although notably not for all the right reasons)
"""

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Note:** What you'll learn throughout this course is that there are **many prompt engineering techniques you can use to derive similar results**. Which techniques you use is up to you and your preference! We encourage you to **experiment to find your own prompt engineering style**.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 3.1 - Math Correction](#exercise-31---math-correction)
"""

"""
### Exercise 3.1 - Math Correction
In some instances, **Claude may struggle with mathematics**, even simple mathematics. Below, Claude incorrectly assesses the math problem as correctly solved, even though there's an obvious arithmetic mistake in the second step. Note that Claude actually catches the mistake when going through step-by-step, but doesn't jump to the conclusion that the overall solution is wrong.

Modify the `PROMPT` and / or the `SYSTEM_PROMPT` to make Claude grade the solution as `incorrectly` solved, rather than correctly solved. 

"""

# System prompt - if you don't want to use a system prompt, you can leave this variable set to an empty string
SYSTEM_PROMPT = ""

# Prompt
PROMPT = """Is this equation solved correctly below?

2x - 3 = 9
2x = 6
x = 3"""

# Get Claude's response
response = get_completion(PROMPT, SYSTEM_PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    if "incorrect" in text or "not correct" in text.lower():
        return True
    else:
        return False

# Print Claude's response and the corresponding grade
print(response)
print("\n--------------------------- GRADING ---------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_3_1_hint; print(exercise_3_1_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a cat."

# Prompt
PROMPT = "In one sentence, what do you think about skateboarding?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a logic bot designed to answer complex logic problems."

# Prompt
PROMPT = "Jack is looking at Anne. Anne is looking at George. Jack is married, George is not, and we don’t know if Anne is married. Is a married person looking at an unmarried person?"

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))



================================================
FILE: Anthropic 1P/04_Separating_Data_and_Instructions.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 4: Separating Data and Instructions

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str, system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

Oftentimes, we don't want to write full prompts, but instead want **prompt templates that can be modified later with additional input data before submitting to Claude**. This might come in handy if you want Claude to do the same thing every time, but the data that Claude uses for its task might be different each time. 

Luckily, we can do this pretty easily by **separating the fixed skeleton of the prompt from variable user input, then substituting the user input into the prompt** before sending the full prompt to Claude. 

Below, we'll walk step by step through how to write a substitutable prompt template, as well as how to substitute in user input.
"""

"""
### Examples

In this first example, we're asking Claude to act as an animal noise generator. Notice that the full prompt submitted to Claude is just the `PROMPT_TEMPLATE` substituted with the input (in this case, "Cow"). Notice that the word "Cow" replaces the `ANIMAL` placeholder via an f-string when we print out the full prompt.

**Note:** You don't have to call your placeholder variable anything in particular in practice. We called it `ANIMAL` in this example, but just as easily, we could have called it `CREATURE` or `A` (although it's generally good to have your variable names be specific and relevant so that your prompt template is easy to understand even without the substitution, just for user parseability). Just make sure that whatever you name your variable is what you use for the prompt template f-string.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why would we want to separate and substitute inputs like this? Well, **prompt templates simplify repetitive tasks**. Let's say you build a prompt structure that invites third party users to submit content to the prompt (in this case the animal whose sound they want to generate). These third party users don't have to write or even see the full prompt. All they have to do is fill in variables.

We do this substitution here using variables and f-strings, but you can also do it with the format() method.

**Note:** Prompt templates can have as many variables as desired!
"""

"""
When introducing substitution variables like this, it is very important to **make sure Claude knows where variables start and end** (vs. instructions or task descriptions). Let's look at an example where there is no separation between the instructions and the substitution variable.

To our human eyes, it is very clear where the variable begins and ends in the prompt template below. However, in the fully substituted prompt, that delineation becomes unclear.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Here, **Claude thinks "Yo Claude" is part of the email it's supposed to rewrite**! You can tell because it begins its rewrite with "Dear Claude". To the human eye, it's clear, particularly in the prompt template where the email begins and ends, but it becomes much less clear in the prompt after substitution.
"""

"""
How do we solve this? **Wrap the input in XML tags**! We did this below, and as you can see, there's no more "Dear Claude" in the output.

[XML tags](https://docs.anthropic.com/claude/docs/use-xml-tags) are angle-bracket tags like `<tag></tag>`. They come in pairs and consist of an opening tag, such as `<tag>`, and a closing tag marked by a `/`, such as `</tag>`. XML tags are used to wrap around content, like this: `<tag>content</tag>`.

**Note:** While Claude can recognize and work with a wide range of separators and delimeters, we recommend that you **use specifically XML tags as separators** for Claude, as Claude was trained specifically to recognize XML tags as a prompt organizing mechanism. Outside of function calling, **there are no special sauce XML tags that Claude has been trained on that you should use to maximally boost your performance**. We have purposefully made Claude very malleable and customizable this way.
"""

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Let's see another example of how XML tags can help us. 

In the following prompt, **Claude incorrectly interprets what part of the prompt is the instruction vs. the input**. It incorrectly considers `Each is about an animal, like rabbits` to be part of the list due to the formatting, when the user (the one filling out the `SENTENCES` variable) presumably did not want that.
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
To fix this, we just need to **surround the user input sentences in XML tags**. This shows Claude where the input data begins and ends despite the misleading hyphen before `Each is about an animal, like rabbits.`
"""

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
**Note:** In the incorrect version of the "Each is about an animal" prompt, we had to include the hyphen to get Claude to respond incorrectly in the way we wanted to for this example. This is an important lesson about prompting: **small details matter**! It's always worth it to **scrub your prompts for typos and grammatical errors**. Claude is sensitive to patterns (in its early years, before finetuning, it was a raw text-prediction tool), and it's more likely to make mistakes when you make mistakes, smarter when you sound smart, sillier when you sound silly, and so on.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 4.1 - Haiku Topic](#exercise-41---haiku-topic)
- [Exercise 4.2 - Dog Question with Typos](#exercise-42---dog-question-with-typos)
- [Exercise 4.3 - Dog Question Part 2](#exercise-42---dog-question-part-2)
"""

"""
### Exercise 4.1 - Haiku Topic
Modify the `PROMPT` so that it's a template that will take in a variable called `TOPIC` and output a haiku about the topic. This exercise is just meant to test your understanding of the variable templating structure with f-strings.
"""

# Variable content
TOPIC = "Pigs"

# Prompt template with a placeholder for the variable content
PROMPT = f""

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("pigs", text.lower()) and re.search("haiku", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_4_1_hint; print(exercise_4_1_hint)

"""
### Exercise 4.2 - Dog Question with Typos
Fix the `PROMPT` by adding XML tags so that Claude produces the right answer. 

Try not to change anything else about the prompt. The messy and mistake-ridden writing is intentional, so you can see how Claude reacts to such mistakes.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_4_2_hint; print(exercise_4_2_hint)

"""
### Exercise 4.3 - Dog Question Part 2
Fix the `PROMPT` **WITHOUT** adding XML tags. Instead, remove only one or two words from the prompt.

Just as with the above exercises, try not to change anything else about the prompt. This will show you what kind of language Claude can parse and understand.
"""

# Variable content
QUESTION = "ar cn brown?"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hia its me i have a q about dogs jkaerjv {QUESTION} jklmvca tx it help me muhch much atx fst fst answer short short tx"

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("brown", text.lower()))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_4_3_hint; print(exercise_4_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Cow"

# Prompt template with a placeholder for the variable content
PROMPT = f"I will tell you the name of an animal. Please respond with the noise that animal makes. {ANIMAL}"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. {EMAIL} <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
EMAIL = "Show up at 6am tomorrow because I'm the CEO and I say so."

# Prompt template with a placeholder for the variable content
PROMPT = f"Yo Claude. <email>{EMAIL}</email> <----- Make this email more polite but don't change anything else about it."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f"""Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
{SENTENCES}"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
SENTENCES = """- I like how cows sound
- This sentence is about spiders
- This sentence may appear to be about dogs but it's actually about pigs"""

# Prompt template with a placeholder for the variable content
PROMPT = f""" Below is a list of sentences. Tell me the second item on the list.

- Each is about an animal, like rabbits.
<sentences>
{SENTENCES}
</sentences>"""

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))



================================================
FILE: Anthropic 1P/05_Formatting_Output_and_Speaking_for_Claude.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 5: Formatting Output and Speaking for Claude

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

# New argument added for prefill text, with a default value of an empty string
def get_completion(prompt: str, system_prompt="", prefill=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

**Claude can format its output in a wide variety of ways**. You just need to ask for it to do so!

One of these ways is by using XML tags to separate out the response from any other superfluous text. You've already learned that you can use XML tags to make your prompt clearer and more parseable to Claude. It turns out, you can also ask Claude to **use XML tags to make its output clearer and more easily understandable** to humans.
"""

"""
### Examples

Remember the 'poem preamble problem' we solved in Chapter 2 by asking Claude to skip the preamble entirely? It turns out we can also achieve a similar outcome by **telling Claude to put the poem in XML tags**.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

"""
Why is this something we'd want to do? Well, having the output in **XML tags allows the end user to reliably get the poem and only the poem by writing a short program to extract the content between XML tags**.

An extension of this technique is to **put the first XML tag in the `assistant` turn. When you put text in the `assistant` turn, you're basically telling Claude that Claude has already said something, and that it should continue from that point onward. This technique is called "speaking for Claude" or "prefilling Claude's response."

Below, we've done this with the first `<haiku>` XML tag. Notice how Claude continues directly from where we left off.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Claude also excels at using other output formatting styles, notably `JSON`. If you want to enforce JSON output (not deterministically, but close to it), you can also prefill Claude's response with the opening bracket, `{`}.
"""

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
Below is an example of **multiple input variables in the same prompt AND output formatting specification, all done using XML tags**.
"""

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
#### Bonus lesson

If you are calling Claude through the API, you can pass the closing XML tag to the `stop_sequences` parameter to get Claude to stop sampling once it emits your desired tag. This can save money and time-to-last-token by eliminating Claude's concluding remarks after it's already given you the answer you care about.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 5.1 - Steph Curry GOAT](#exercise-51---steph-curry-goat)
- [Exercise 5.2 - Two Haikus](#exercise-52---two-haikus)
- [Exercise 5.3 - Two Haikus, Two Animals](#exercise-53---two-haikus-two-animals)
"""

"""
### Exercise 5.1 - Steph Curry GOAT
Forced to make a choice, Claude designates Michael Jordan as the best basketball player of all time. Can we get Claude to pick someone else?

Change the `PREFILL` variable to **compell Claude to make a detailed argument that the best basketball player of all time is Stephen Curry**. Try not to change anything except `PREFILL` as that is the focus of this exercise.
"""

# Prompt template with a placeholder for the variable content
PROMPT = f"Who is the best basketball player of all time? Please choose one specific player."

# Prefill for Claude's response
PREFILL = ""

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("Warrior", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_5_1_hint; print(exercise_5_1_hint)

"""
### Exercise 5.2 - Two Haikus
Modify the `PROMPT` below using XML tags so that Claude writes two haikus about the animal instead of just one. It should be clear where one poem ends and the other begins.
"""

# Variable content
ANIMAL = "cats"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Get Claude's response
response = get_completion(PROMPT, prefill=PREFILL)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(
        (re.search("cat", text.lower()) and re.search("<haiku>", text))
        and (text.count("\n") + 1) > 5
    )

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_5_2_hint; print(exercise_5_2_hint)

"""
### Exercise 5.3 - Two Haikus, Two Animals
Modify the `PROMPT` below so that **Claude produces two haikus about two different animals**. Use `{ANIMAL1}` as a stand-in for the first substitution, and `{ANIMAL2}` as a stand-in for the second substitution.
"""

# First input variable
ANIMAL1 = "Cat"

# Second input variable
ANIMAL2 = "Dog"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL1}. Put it in <haiku> tags."

# Get Claude's response
response = get_completion(PROMPT)

# Function to grade exercise correctness
def grade_exercise(text):
    return bool(re.search("tail", text.lower()) and re.search("cat", text.lower()) and re.search("<haiku>", text))

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(response)
print("\n------------------------------------------ GRADING ------------------------------------------")
print("This exercise has been correctly solved:", grade_exercise(response))

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_5_3_hint; print(exercise_5_3_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Variable content
ANIMAL = "Rabbit"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print(PROMPT)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Put it in <haiku> tags."

# Prefill for Claude's response
PREFILL = "<haiku>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# Variable content
ANIMAL = "Cat"

# Prompt template with a placeholder for the variable content
PROMPT = f"Please write a haiku about {ANIMAL}. Use JSON format with the keys as \"first_line\", \"second_line\", and \"third_line\"."

# Prefill for Claude's response
PREFILL = "{"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

# First input variable
EMAIL = "Hi Zack, just pinging you for a quick update on that prompt you were supposed to write."

# Second input variable
ADJECTIVE = "olde english"

# Prompt template with a placeholder for the variable content
PROMPT = f"Hey Claude. Here is an email: <email>{EMAIL}</email>. Make this email more {ADJECTIVE}. Write the new version in <{ADJECTIVE}_email> XML tags."

# Prefill for Claude's response (now as an f-string with a variable)
PREFILL = f"<{ADJECTIVE}_email>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN")
print(PROMPT)
print("\nASSISTANT TURN")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: Anthropic 1P/06_Precognition_Thinking_Step_by_Step.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 6: Precognition (Thinking Step by Step)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str, system_prompt="", prefill=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

If someone woke you up and immediately started asking you several complicated questions that you had to respond to right away, how would you do? Probably not as good as if you were given time to **think through your answer first**. 

Guess what? Claude is the same way.

**Giving Claude time to think step by step sometimes makes Claude more accurate**, particularly for complex tasks. However, **thinking only counts when it's out loud**. You cannot ask Claude to think but output only the answer - in this case, no thinking has actually occurred.
"""

"""
### Examples

In the prompt below, it's clear to a human reader that the second sentence belies the first. But **Claude takes the word "unrelated" too literally**.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
To improve Claude's response, let's **allow Claude to think things out first before answering**. We do that by literally spelling out the steps that Claude should take in order to process and think through its task. Along with a dash of role prompting, this empowers Claude to understand the review more deeply.
"""

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

"""
**Claude is sometimes sensitive to ordering**. This example is on the frontier of Claude's ability to understand nuanced text, and when we swap the order of the arguments from the previous example so that negative is first and positive is second, this changes Claude's overall assessment to positive.

In most situations (but not all, confusingly enough), **Claude is more likely to choose the second of two options**, possibly because in its training data from the web, second options were more likely to be correct.
"""

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

"""
**Letting Claude think can shift Claude's answer from incorrect to correct**. It's that simple in many cases where Claude makes mistakes!

Let's go through an example where Claude's answer is incorrect to see how asking Claude to think can fix that.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

"""
Let's fix this by asking Claude to think step by step, this time in `<brainstorm>` tags.
"""

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 6.1 - Classifying Emails](#exercise-61---classifying-emails)
- [Exercise 6.2 - Email Classification Formatting](#exercise-62---email-classification-formatting)
"""

"""
### Exercise 6.1 - Classifying Emails
In this exercise, we'll be instructing Claude to sort emails into the following categories:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)

For the first part of the exercise, change the `PROMPT` to **make Claude output the correct classification and ONLY the classification**. Your answer needs to **include the letter (A - D) of the correct choice, with the parentheses, as well as the name of the category**.

Refer to the comments beside each email in the `EMAILS` list to know which category that email should be classified under.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "A\) P",
    "B": "B\) B",
    "C": "C\) B",
    "D": "D\) O"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_6_1_hint; print(exercise_6_1_hint)

"""
Still stuck? Run the cell below for an example solution.						
"""

from hints import exercise_6_1_solution; print(exercise_6_1_solution)

"""
### Exercise 6.2 - Email Classification Formatting
In this exercise, we're going to refine the output of the above prompt to yield an answer formatted exactly how we want it. 

Use your favorite output formatting technique to make Claude wrap JUST the letter of the correct classification in `<answer></answer>` tags. For instance, the answer to the first email should contain the exact string `<answer>B</answer>`.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response, if any
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Dictionary of string values for each category to be used for regex grading
REGEX_CATEGORIES = {
    "A": "<answer>A</answer>",
    "B": "<answer>B</answer>",
    "C": "<answer>C</answer>",
    "D": "<answer>D</answer>"
}

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(REGEX_CATEGORIES[ans], response)) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_6_2_hint; print(exercise_6_2_hint)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = """Is this movie review sentiment positive or negative?

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since the year 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# System prompt
SYSTEM_PROMPT = "You are a savvy reader of movie reviews."

# Prompt
PROMPT = """Is this review sentiment positive or negative? First, write the best arguments for each side in <positive-argument> and <negative-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. In totally unrelated news, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT, SYSTEM_PROMPT))

# Prompt
PROMPT = """Is this review sentiment negative or positive? First write the best arguments for each side in <negative-argument> and <positive-argument> XML tags, then answer.

This movie blew my mind with its freshness and originality. Unrelatedly, I have been living under a rock since 1900."""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956."

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = "Name a famous movie starring an actor who was born in the year 1956. First brainstorm about some actors and their birth years in <brainstorm> tags, then give your answer."

# Print Claude's response
print(get_completion(PROMPT))



================================================
FILE: Anthropic 1P/07_Using_Examples_Few-Shot_Prompting.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Chapter 7: Using Examples (Few-Shot Prompting)

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

def get_completion(prompt: str, system_prompt="", prefill=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=[
          {"role": "user", "content": prompt},
          {"role": "assistant", "content": prefill}
        ]
    )
    return message.content[0].text

"""
---

## Lesson

**Giving Claude examples of how you want it to behave (or how you want it not to behave) is extremely effective** for:
- Getting the right answer
- Getting the answer in the right format

This sort of prompting is also called "**few shot prompting**". You might also encounter the phrase "zero-shot" or "n-shot" or "one-shot". The number of "shots" refers to how many examples are used within the prompt.
"""

"""
### Examples

Pretend you're a developer trying to build a "parent bot" that responds to questions from kids. **Claude's default response is quite formal and robotic**. This is going to break a child's heart.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

"""
You could take the time to describe your desired tone, but it's much easier just to **give Claude a few examples of ideal responses**.
"""

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

"""
In the following formatting example, we could walk Claude step by step through a set of formatting instructions on how to extract names and professions and then format them exactly the way we want, or we could just **provide Claude with some correctly-formatted examples and Claude can extrapolate from there**. Note the `<individuals>` in the `assistant` turn to start Claude off on the right foot.
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))

"""
If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 7.1 - Email Formatting via Examples](#exercise-71---email-formatting-via-examples)
"""

"""
### Exercise 7.1 - Email Formatting via Examples
We're going to redo Exercise 6.2, but this time, we're going to edit the `PROMPT` to use "few-shot" examples of emails + proper classification (and formatting) to get Claude to output the correct answer. We want the *last* letter of Claude's output to be the letter of the category.

Refer to the comments beside each email in the `EMAILS` list if you forget which letter category is correct for each email.

Remember that these are the categories for the emails:										
- (A) Pre-sale question
- (B) Broken or defective item
- (C) Billing question
- (D) Other (please explain)								
"""

# Prompt template with a placeholder for the variable content
PROMPT = """Please classify this email as either green or blue: {email}"""

# Prefill for Claude's response
PREFILL = ""

# Variable content stored as a list
EMAILS = [
    "Hi -- My Mixmaster4000 is producing a strange noise when I operate it. It also smells a bit smoky and plasticky, like burning electronics.  I need a replacement.", # (B) Broken or defective item
    "Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?", # (A) Pre-sale question OR (D) Other (please explain)
    "I HAVE BEEN WAITING 4 MONTHS FOR MY MONTHLY CHARGES TO END AFTER CANCELLING!!  WTF IS GOING ON???", # (C) Billing question
    "How did I get here I am not good with computer.  Halp." # (D) Other (please explain)
]

# Correct categorizations stored as a list of lists to accommodate the possibility of multiple correct categorizations per email
ANSWERS = [
    ["B"],
    ["A","D"],
    ["C"],
    ["D"]
]

# Iterate through list of emails
for i,email in enumerate(EMAILS):
    
    # Substitute the email text into the email placeholder variable
    formatted_prompt = PROMPT.format(email=email)
   
    # Get Claude's response
    response = get_completion(formatted_prompt, prefill=PREFILL)

    # Grade Claude's response
    grade = any([bool(re.search(ans, response[-1])) for ans in ANSWERS[i]])
    
    # Print Claude's response
    print("--------------------------- Full prompt with variable substutions ---------------------------")
    print("USER TURN")
    print(formatted_prompt)
    print("\nASSISTANT TURN")
    print(PREFILL)
    print("\n------------------------------------- Claude's response -------------------------------------")
    print(response)
    print("\n------------------------------------------ GRADING ------------------------------------------")
    print("This exercise has been correctly solved:", grade, "\n\n\n\n\n\n")

"""
❓ If you want a hint, run the cell below!
"""

from hints import exercise_7_1_hint; print(exercise_7_1_hint)

"""
Still stuck? Run the cell below for an example solution.
"""

from hints import exercise_7_1_solution; print(exercise_7_1_solution)

"""
### Congrats!

If you've solved all exercises up until this point, you're ready to move to the next chapter. Happy prompting!
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Prompt
PROMPT = "Will Santa bring me presents on Christmas?"

# Print Claude's response
print(get_completion(PROMPT))

# Prompt
PROMPT = """Please complete the conversation by writing the next line, speaking as "A".
Q: Is the tooth fairy real?
A: Of course, sweetie. Wrap up your tooth and put it under your pillow tonight. There might be something waiting for you in the morning.
Q: Will Santa bring me presents on Christmas?"""

# Print Claude's response
print(get_completion(PROMPT))

# Prompt template with a placeholder for the variable content
PROMPT = """Silvermist Hollow, a charming village, was home to an extraordinary group of individuals.
Among them was Dr. Liam Patel, a neurosurgeon who revolutionized surgical techniques at the regional medical center.
Olivia Chen was an innovative architect who transformed the village's landscape with her sustainable and breathtaking designs.
The local theater was graced by the enchanting symphonies of Ethan Kovacs, a professionally-trained musician and composer.
Isabella Torres, a self-taught chef with a passion for locally sourced ingredients, created a culinary sensation with her farm-to-table restaurant, which became a must-visit destination for food lovers.
These remarkable individuals, each with their distinct talents, contributed to the vibrant tapestry of life in Silvermist Hollow.
<individuals>
1. Dr. Liam Patel [NEUROSURGEON]
2. Olivia Chen [ARCHITECT]
3. Ethan Kovacs [MISICIAN AND COMPOSER]
4. Isabella Torres [CHEF]
</individuals>

At the heart of the town, Chef Oliver Hamilton has transformed the culinary scene with his farm-to-table restaurant, Green Plate. Oliver's dedication to sourcing local, organic ingredients has earned the establishment rave reviews from food critics and locals alike.
Just down the street, you'll find the Riverside Grove Library, where head librarian Elizabeth Chen has worked diligently to create a welcoming and inclusive space for all. Her efforts to expand the library's offerings and establish reading programs for children have had a significant impact on the town's literacy rates.
As you stroll through the charming town square, you'll be captivated by the beautiful murals adorning the walls. These masterpieces are the work of renowned artist, Isabella Torres, whose talent for capturing the essence of Riverside Grove has brought the town to life.
Riverside Grove's athletic achievements are also worth noting, thanks to former Olympic swimmer-turned-coach, Marcus Jenkins. Marcus has used his experience and passion to train the town's youth, leading the Riverside Grove Swim Team to several regional championships.
<individuals>
1. Oliver Hamilton [CHEF]
2. Elizabeth Chen [LIBRARIAN]
3. Isabella Torres [ARTIST]
4. Marcus Jenkins [COACH]
</individuals>

Oak Valley, a charming small town, is home to a remarkable trio of individuals whose skills and dedication have left a lasting impact on the community.
At the town's bustling farmer's market, you'll find Laura Simmons, a passionate organic farmer known for her delicious and sustainably grown produce. Her dedication to promoting healthy eating has inspired the town to embrace a more eco-conscious lifestyle.
In Oak Valley's community center, Kevin Alvarez, a skilled dance instructor, has brought the joy of movement to people of all ages. His inclusive dance classes have fostered a sense of unity and self-expression among residents, enriching the local arts scene.
Lastly, Rachel O'Connor, a tireless volunteer, dedicates her time to various charitable initiatives. Her commitment to improving the lives of others has been instrumental in creating a strong sense of community within Oak Valley.
Through their unique talents and unwavering dedication, Laura, Kevin, and Rachel have woven themselves into the fabric of Oak Valley, helping to create a vibrant and thriving small town."""

# Prefill for Claude's response
PREFILL = "<individuals>"

# Print Claude's response
print("--------------------------- Full prompt with variable substutions ---------------------------")
print("USER TURN:")
print(PROMPT)
print("\nASSISTANT TURN:")
print(PREFILL)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(PROMPT, prefill=PREFILL))



================================================
FILE: Anthropic 1P/10.1_Appendix_Chaining Prompts.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.1: Chaining Prompts

- [Lesson](#lesson)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY & MODEL_NAME variables from the IPython store
%store -r API_KEY
%store -r MODEL_NAME

client = anthropic.Anthropic(api_key=API_KEY)

# Has been rewritten to take in a messages list of arbitrary length
def get_completion(messages, system_prompt=""):
    message = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=messages
    )
    return message.content[0].text

"""
---

## Lesson

The saying goes, "Writing is rewriting." It turns out, **Claude can often improve the accuracy of its response when asked to do so**!

There are many ways to prompt Claude to "think again". The ways that feel natural to ask a human to double check their work will also generally work for Claude. (Check out our [prompt chaining documentation](https://docs.anthropic.com/claude/docs/chain-prompts) for further examples of when and how to use prompt chaining.)
"""

"""
### Examples

In this example, we ask Claude to come up with ten words... but one or more of them isn't a real word.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
**Asking Claude to make its answer more accurate** fixes the error! 

Below, we've pulled down Claude's incorrect response from above and added another turn to the conversation asking Claude to fix its previous answer.
"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
But is Claude revising its answer just because we told it to? What if we start off with a correct answer already? Will Claude lose its confidence? Here, we've placed a correct response in the place of `first_response` and asked it to double check again.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
You may notice that if you generate a respnse from the above block a few times, Claude leaves the words as is most of the time, but still occasionally changes the words even though they're all already correct. What can we do to mitigate this? Per Chapter 8, we can give Claude an out! Let's try this one more time.
"""

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Try generating responses from the above code a few times to see that Claude is much better at sticking to its guns now.

You can also use prompt chaining to **ask Claude to make its responses better**. Below, we asked Claude to first write a story, and then improve the story it wrote. Your personal tastes may vary, but many might agree that Claude's second version is better.

First, let's generate Claude's first version of the story.
"""

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

"""
Now let's have Claude improve on its first draft.
"""

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
This form of substitution is very powerful. We've been using substitution placeholders to pass in lists, words, Claude's former responses, and so on. You can also **use substitution to do what we call "function calling," which is asking Claude to perform some function, and then taking the results of that function and asking Claude to do even more afterward with the results**. It works like any other substitution. More on this in the next appendix.

Below is one more example of taking the results of one call to Claude and plugging it into another, longer call. Let's start with the first prompt (which includes prefilling Claude's response this time).
"""

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

"""
Let's pass this list of names into another prompt.
"""

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

"""
Now that you've learned about prompt chaining, head over to Appendix 10.2 to learn how to implement function calling using prompt chaining.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

# Initial prompt
first_user = "Name ten words that all end with the exact letters 'ab'."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = "Name ten words that all end with the exact letters 'ab'."

first_response = """Here are 10 words that end with the letters 'ab':

1. Cab
2. Dab
3. Grab
4. Gab
5. Jab
6. Lab
7. Nab
8. Slab
9. Tab
10. Blab"""

second_user = "Please find replacements for all 'words' that are not real words. If all the words are real words, return the original list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

# Initial prompt
first_user = "Write a three-sentence short story about a girl who likes to run."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print(first_response)

second_user = "Make the story better."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))

first_user = """Find all names from the below text:

"Hey, Jesse. It's me, Erin. I'm calling about the party that Joey is throwing tomorrow. Keisha said she would come and I think Mel will be there too."""

prefill = "<names>"

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill
    
    }
]

# Store and print Claude's response
first_response = get_completion(messages)
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(first_response)

second_user = "Alphabetize the list."

# API messages array
messages = [
    {
        "role": "user",
        "content": first_user
    
    },
    {
        "role": "assistant",
        "content": prefill + "\n" + first_response
    
    },
    {
        "role": "user",
        "content": second_user
    
    }
]

# Print Claude's response
print("------------------------ Full messsages array with variable substutions ------------------------")
print(messages)
print("\n------------------------------------- Claude's response -------------------------------------")
print(get_completion(messages))



================================================
FILE: Anthropic 1P/10.2_Appendix_Tool Use.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.2: Tool Use

- [Lesson](#lesson)
- [Exercises](#exercises)
- [Example Playground](#example-playground)

## Setup

Run the following setup cell to load your API key and establish the `get_completion` helper function.
"""

!pip install anthropic

# Import python's built-in regular expression library
import re
import anthropic

# Retrieve the API_KEY variable from the IPython store
%store -r API_KEY

client = anthropic.Anthropic(api_key=API_KEY)

# Rewrittten to call Claude 3 Sonnet, which is generally better at tool use, and include stop_sequences
def get_completion(messages, system_prompt="", prefill="",stop_sequences=None):
    message = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=2000,
        temperature=0.0,
        system=system_prompt,
        messages=messages,
        stop_sequences=stop_sequences
    )
    return message.content[0].text

"""
---

## Lesson

While it might seem conceptually complex at first, tool use, a.k.a. function calling, is actually quite simple! You already know all the skills necessary to implement tool use, which is really just a combination of substitution and prompt chaining.

In previous substitution exercises, we substituted text into prompts. With tool use, we substitute tool or function results into prompts. Claude can't literally call or access tools and functions. Instead, we have Claude:
1. Output the tool name and arguments it wants to call
2. Halt any further response generation while the tool is called
3. Then we reprompt with the appended tool results
"""

"""
Function calling is useful because it expands Claude's capabilities and enables Claude to handle much more complex, multi-step tasks.
Some examples of functions you can give Claude:
- Calculator
- Word counter
- SQL database querying and data retrieval
- Weather API
"""

"""
You can get Claude to do tool use by combining these two elements:

1. A system prompt, in which we give Claude an explanation of the concept of tool use as well as a detailed descriptive list of the tools it has access to
2. The control logic with which to orchestrate and execute Claude's tool use requests
"""

"""
### Tool use roadmap

*This lesson teaches our current tool use format. However, we will be updating and improving tool use functionality in the near future, including:*
* *A more streamlined format for function definitions and calls*
* *More robust error handilgj and edge case coverage*
* *Tighter integration with the rest of our API*
* *Better reliability and performance, especially for more complex tool use tasks*
"""

"""
### Examples

To enable tool use in Claude, we start with the system prompt. In this special tool use system prompt, wet tell Claude:
* The basic premise of tool use and what it entails
* How Claude can call and use the tools it's been given
* A detailed list of tools it has access to in this specific scenario 

Here's the first part of the system prompt, explaining tool use to Claude. This part of the system prompt is generalizable across all instances of prompting Claude for tool use. The tool calling structure we're giving Claude (`<function_calls> [...] </function_calls>`) is a structure Claude has been specifically trained to use, so we recommend that you stick with this.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

"""
Here's the second part of the system prompt, which defines the exact tools Claude has access to in this specific situation. In this example, we will be giving Claude a calculator tool, which takes three parameters: two operands and an operator. 

Then we combine the two parts of the system prompt.
"""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

"""
Now we can give Claude a question that requires use of the `calculator` tool. We will use `<function_calls\>` in `stop_sequences` to detect if and when Claude calls the function.
"""

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Now, we can extract out the parameters from Claude's function call and actually run the function on Claude's behalf.

First we'll define the function's code.
"""

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

"""
Then we'll extract the parameters from Claude's function call response. If all the parameters exist, we run the calculator tool.
"""

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

"""
Now that we have a result, we have to properly format that result so that when we pass it back to Claude, Claude understands what tool that result is in relation to. There is a set format for this that Claude has been trained to recognize:
```
<function_results>
<result>
<tool_name>{TOOL_NAME}</tool_name>
<stdout>
{TOOL_RESULT}
</stdout>
</result>
</function_results>
```

Run the cell below to format the above tool result into this structure.
"""

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

"""
Now all we have to do is send this result back to Claude by appending the result to the same message chain as before, and we're good!
"""

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

"""
Congratulations on running an entire tool use chain end to end!

Now what if we give Claude a question that doesn't that doesn't require using the given tool at all?
"""

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

"""
Success! As you can see, Claude knew not to call the function when it wasn't needed.

If you would like to experiment with the lesson prompts without changing any content above, scroll all the way to the bottom of the lesson notebook to visit the [**Example Playground**](#example-playground).
"""

"""
---

## Exercises
- [Exercise 10.2.1 - SQL](#exercise-1021---SQL)
"""

"""
### Exercise 10.2.1 - SQL
In this exercise, you'll be writing a tool use prompt for querying and writing to the world's smallest "database". Here's the initialized database, which is really just a dictionary.
"""

db = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com"}
    ],
    "products": [
        {"id": 1, "name": "Widget", "price": 9.99},
        {"id": 2, "name": "Gadget", "price": 14.99},
        {"id": 3, "name": "Doohickey", "price": 19.99}
    ]
}

"""
And here is the code for the functions that write to and from the database.
"""

def get_user(user_id):
    for user in db["users"]:
        if user["id"] == user_id:
            return user
    return None

def get_product(product_id):
    for product in db["products"]:
        if product["id"] == product_id:
            return product
    return None

def add_user(name, email):
    user_id = len(db["users"]) + 1
    user = {"id": user_id, "name": name, "email": email}
    db["users"].append(user)
    return user

def add_product(name, price):
    product_id = len(db["products"]) + 1
    product = {"id": product_id, "name": name, "price": price}
    db["products"].append(product)
    return product

"""
To solve the exercise, start by defining a system prompt like `system_prompt_tools_specific_tools` above. Make sure to include the name and description of each tool, along with the name and type and description of each parameter for each function. We've given you some starting scaffolding below.
"""

system_prompt_tools_specific_tools_sql = """
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools_sql

"""
When you're ready, you can try out your tool definition system prompt on the examples below. Just run the below cell!
"""

examples = [
    "Add a user to the database named Deborah.",
    "Add a product to the database named Thingo",
    "Tell me the name of User 2",
    "Tell me the name of Product 3"
]

for example in examples:
    message = {
        "role": "user",
        "content": example
    }

    # Get & print Claude's response
    function_calling_response = get_completion([message], system_prompt=system_prompt, stop_sequences=stop_sequences)
    print(example, "\n----------\n\n", function_calling_response, "\n*********\n*********\n*********\n\n")

"""
If you did it right, the function calling messages should call the `add_user`, `add_product`, `get_user`, and `get_product` functions correctly.

For extra credit, add some code cells and write parameter-parsing code. Then call the functions with the parameters Claude gives you to see the state of the "database" after the call.
"""

"""
❓ If you want to see a possible solution, run the cell below!
"""

from hints import exercise_10_2_1_solution; print(exercise_10_2_1_solution)

"""
### Congrats!

Congratulations on learning tool use and function calling! Head over to the last appendix section if you would like to learn more about search & RAG.
"""

"""
---

## Example Playground

This is an area for you to experiment freely with the prompt examples shown in this lesson and tweak prompts to see how it may affect Claude's responses.
"""

system_prompt_tools_general_explanation = """You have access to a set of functions you can use to answer the user's question. This includes access to a
sandboxed computing environment. You do NOT currently have the ability to inspect files or interact with external
resources, except by invoking the below functions.

You can invoke one or more functions by writing a "<function_calls>" block like the following as part of your
reply to the user:
<function_calls>
<invoke name="$FUNCTION_NAME">
<antml:parameter name="$PARAMETER_NAME">$PARAMETER_VALUE</parameter>
...
</invoke>
<nvoke name="$FUNCTION_NAME2">
...
</invoke>
</function_calls>

String and scalar parameters should be specified as is, while lists and objects should use JSON format. Note that
spaces for string values are not stripped. The output is not expected to be valid XML and is parsed with regular
expressions.

The output and/or any errors will appear in a subsequent "<function_results>" block, and remain there as part of
your reply to the user.
You may then continue composing the rest of your reply to the user, respond to any errors, or make further function
calls as appropriate.
If a "<function_results>" does NOT appear after your function calls, then they are likely malformatted and not
recognized as a call."""

system_prompt_tools_specific_tools = """Here are the functions available in JSONSchema format:
<tools>
<tool_description>
<tool_name>calculator</tool_name>
<description>
Calculator function for doing basic arithmetic.
Supports addition, subtraction, multiplication
</description>
<parameters>
<parameter>
<name>first_operand</name>
<type>int</type>
<description>First operand (before the operator)</description>
</parameter>
<parameter>
<name>second_operand</name>
<type>int</type>
<description>Second operand (after the operator)</description>
</parameter>
<parameter>
<name>operator</name>
<type>str</type>
<description>The operation to perform. Must be either +, -, *, or /</description>
</parameter>
</parameters>
</tool_description>
</tools>
"""

system_prompt = system_prompt_tools_general_explanation + system_prompt_tools_specific_tools

multiplication_message = {
    "role": "user",
    "content": "Multiply 1,984,135 by 9,343,116"
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)

def do_pairwise_arithmetic(num1, num2, operation):
    if operation == '+':
        return num1 + num2
    elif operation == "-":
        return num1 - num2
    elif operation == "*":
        return num1 * num2
    elif operation == "/":
        return num1 / num2
    else:
        return "Error: Operation not supported."

def find_parameter(message, parameter_name):
    parameter_start_string = f"name=\"{parameter_name}\">"
    start = message.index(parameter_start_string)
    if start == -1:
        return None
    if start > 0:
        start = start + len(parameter_start_string)
        end = start
        while message[end] != "<":
            end += 1
    return message[start:end]

first_operand = find_parameter(function_calling_response, "first_operand")
second_operand = find_parameter(function_calling_response, "second_operand")
operator = find_parameter(function_calling_response, "operator")

if first_operand and second_operand and operator:
    result = do_pairwise_arithmetic(int(first_operand), int(second_operand), operator)
    print("---------------- RESULT ----------------")
    print(f"{result:,}")

def construct_successful_function_run_injection_prompt(invoke_results):
    constructed_prompt = (
        "<function_results>\n"
        + '\n'.join(
            f"<result>\n<tool_name>{res['tool_name']}</tool_name>\n<stdout>\n{res['tool_result']}\n</stdout>\n</result>"
            for res in invoke_results
        ) + "\n</function_results>"
    )

    return constructed_prompt

formatted_results = [{
    'tool_name': 'do_pairwise_arithmetic',
    'tool_result': result
}]
function_results = construct_successful_function_run_injection_prompt(formatted_results)
print(function_results)

full_first_response = function_calling_response + "</function_calls>"

# Construct the full conversation
messages = [multiplication_message,
{
    "role": "assistant",
    "content": full_first_response
},
{
    "role": "user",
    "content": function_results
}]
   
# Print Claude's response
final_response = get_completion(messages, system_prompt=system_prompt, stop_sequences=stop_sequences)
print("------------- FINAL RESULT -------------")
print(final_response)

non_multiplication_message = {
    "role": "user",
    "content": "Tell me the capital of France."
}

stop_sequences = ["</function_calls>"]

# Get Claude's response
function_calling_response = get_completion([non_multiplication_message], system_prompt=system_prompt, stop_sequences=stop_sequences)
print(function_calling_response)



================================================
FILE: Anthropic 1P/10.3_Appendix_Search & Retrieval.ipynb
================================================
# Jupyter notebook converted to Python script.

"""
# Appendix 10.3: Search & Retrieval

Did you know you can use Claude to **search through Wikipedia for you**? Claude can find and retrieve articles, at which point you can also use Claude to summarize and synthesize them, write novel content from what it found, and much more. And not just Wikipedia! You can also search over your own docs, whether stored as plain text or embedded in a vector datastore.

See our [RAG cookbook examples](https://github.com/anthropics/anthropic-cookbook/blob/main/third_party/Wikipedia/wikipedia-search-cookbook.ipynb) to learn how to supplement Claude's knowledge and improve the accuracy and relevance of Claude's responses with data retrieved from vector databases, Wikipedia, the internet, and more. There, you can also learn about how to use certain [embeddings](https://docs.anthropic.com/claude/docs/embeddings) and vector database tools.

If you are interested in learning about advanced RAG architectures using Claude, check out our [Claude 3 technical presentation slides on RAG architectures](https://docs.google.com/presentation/d/1zxkSI7lLUBrZycA-_znwqu8DDyVhHLkQGScvzaZrUns/edit#slide=id.g2c736259dac_63_782).
"""



================================================
FILE: Anthropic 1P/hints.py
================================================
exercise_1_1_hint = """The grading function in this exercise is looking for an answer that contains the exact Arabic numerals "1", "2", and "3".
You can often get Claude to do what you want simply by asking."""

exercise_1_2_hint = """The grading function in this exercise is looking for answers that contain "soo" or "giggles".
There are many ways to solve this, just by asking!"""

exercise_2_1_hint ="""The grading function in this exercise is looking for any answer that includes the word "hola".
Ask Claude to reply in Spanish like you would when speaking with a human. It's that simple!"""

exercise_2_2_hint = """The grading function in this exercise is looking for EXACTLY "Michael Jordan".
How would you ask another human to do this? Reply with no other words? Reply with only the name and nothing else? There are several ways to approach this answer."""

exercise_2_3_hint = """The grading function in this cell is looking for a response that is equal to or greater than 800 words.
Because LLMs aren't great at counting words yet, you may have to overshoot your target."""

exercise_3_1_hint = """The grading function in this exercise is looking for an answer that includes the words "incorrect" or "not correct".
Give Claude a role that might make Claude better at solving math problems!"""

exercise_4_1_hint = """The grading function in this exercise is looking for a solution that includes the words "haiku" and "pig".
Don't forget to include the exact phrase "{TOPIC}" wherever you want the topic to be substituted in. Changing the "TOPIC" variable value should make Claude write a haiku about a different topic."""

exercise_4_2_hint = """The grading function in this exercise is looking for a response that includes the word "brown".
If you surround "{QUESTION}" in XML tags, how does that change Claude's response?"""

exercise_4_3_hint = """The grading function in this exercise is looking for a response that includes the word "brown".
Try removing one word or section of characters at a time, starting with the parts that make the least sense. Doing this one word at a time will also help you see just how much Claude can or can't parse and understand."""

exercise_5_1_hint = """The grading function for this exercise is looking for a response that includes the word "Warrior".
Write more words in Claude's voice to steer Claude to act the way you want it to. For instance, instead of "Stephen Curry is the best because," you could write "Stephen Curry is the best and here are three reasons why. 1:"""

exercise_5_2_hint = """The grading function looks for a response of over 5 lines in length that includes the words "cat" and "<haiku>".
Start simple. Currently, the prompt asks Claude for one haiku. You can change that and ask for two (or even more). Then if you run into formatting issues, change your prompt to fix that after you've already gotten Claude to write more than one haiku."""

exercise_5_3_hint = """The grading function in this exercise is looking for a response that contains the words "tail", "cat", and "<haiku>".
It's helpful to break this exercise down to several steps.								
1.	Modify the initial prompt template so that Claude writes two poems.							
2.	Give Claude indicators as to what the poems will be about, but instead of writing in the subjects directly (e.g., dog, cat, etc.), replace those subjects with the keywords "{ANIMAL1}" and "{ANIMAL2}".							
3.	Run the prompt and make sure that the full prompt with variable substitutions has all the words correctly substituted. If not, check to make sure your {bracket} tags are spelled correctly and formatted correctly with single moustache brackets."""

exercise_6_1_hint = """The grading function in this exercise is looking for the correct categorization letter + the closing parentheses and the first letter of the name of the category, such as "C) B" or "B) B" etc.
Let's take this exercise step by step:										
1.	How will Claude know what categories you want to use? Tell it! Include the four categories you want directly in the prompt. Be sure to include the parenthetical letters as well for easy classification. Feel free to use XML tags to organize your prompt and make clear to Claude where the categories begin and end.									
2.	Try to cut down on superfluous text so that Claude immediately answers with the classification and ONLY the classification. There are several ways to do this, from speaking for Claude (providing anything from the beginning of the sentence to a single open parenthesis so that Claude knows you want the parenthetical letter as the first part of the answer) to telling Claude that you want the classification and only the classification, skipping the preamble.
Refer to Chapters 2 and 5 if you want a refresher on these techniques.							
3.	Claude may still be incorrectly categorizing or not including the names of the categories when it answers. Fix this by telling Claude to include the full category name in its answer.)								
4.	Be sure that you still have {email} somewhere in your prompt template so that we can properly substitute in emails for Claude to evaluate."""

exercise_6_1_solution = """
USER TURN
Please classify this email into the following categories: {email}

Do not include any extra words except the category.

<categories>
(A) Pre-sale question
(B) Broken or defective item
(C) Billing question
(D) Other (please explain)
</categories>

ASSISTANT TURN
(
"""

exercise_6_2_hint = """The grading function in this exercise is looking for only the correct letter wrapped in <answer> tags, such as "<answer>B</answer>". The correct categorization letters are the same as in the above exercise.
Sometimes the simplest way to go about this is to give Claude an example of how you want its output to look. Just don't forget to wrap your example in <example></example> tags! And don't forget that if you prefill Claude's response with anything, Claude won't actually output that as part of its response."""

exercise_7_1_hint = """You're going to have to write some example emails and classify them for Claude (with the exact formatting you want). There are multiple ways to do this. Here are some guidelines below.										
1.	Try to have at least two example emails. Claude doesn't need an example for all categories, and the examples don't have to be long. It's more helpful to have examples for whatever you think the trickier categories are (which you were asked to think about at the bottom of Chapter 6 Exercise 1). XML tags will help you separate out your examples from the rest of your prompt, although it's unnecessary.									
2.	Make sure your example answer formatting is exactly the format you want Claude to use, so Claude can emulate the format as well. This format should make it so that Claude's answer ends in the letter of the category. Wherever you put the {email} placeholder, make sure that it's formatted exactly like your example emails.									
3.	Make sure you still have the categories listed within the prompt itself, otherwise Claude won't know what categories to reference, as well as {email} as a placeholder for substitution."""

exercise_7_1_solution = """
USER TURN
Please classify emails into the following categories, and do not include explanations: 
<categories>
(A) Pre-sale question
(B) Broken or defective item
(C) Billing question
(D) Other (please explain)
</categories>

Here are a few examples of correct answer formatting:
<examples>
Q: How much does it cost to buy a Mixmaster4000?
A: The correct category is: A

Q: My Mixmaster won't turn on.
A: The correct category is: B

Q: Please remove me from your mailing list.
A: The correct category is: D
</examples>

Here is the email for you to categorize: {email}

ASSISTANT TURN
The correct category is:
"""
exercise_8_1_hint = """The grading function in this exercise is looking for a response that contains the phrase "I do not", "I don't", or "Unfortunately".
What should Claude do if it doesn't know the answer?"""

exercise_8_2_hint = """The grading function in this exercise is looking for a response that contains the phrase "49-fold".
Make Claude show its work and thought process first by extracting relevant quotes and seeing whether or not the quotes provide sufficient evidence. Refer back to the Chapter 8 Lesson if you want a refresher."""

exercise_9_1_solution = """
You are a master tax acountant. Your task is to answer user questions using any provided reference documentation.

Here is the material you should use to answer the user's question:
<docs>
{TAX_CODE}
</docs>

Here is an example of how to respond:
<example>
<question>
What defines a "qualified" employee?
</question>
<answer>
<quotes>For purposes of this subsection—
(A)In general
The term "qualified employee" means any individual who—
(i)is not an excluded employee, and
(ii)agrees in the election made under this subsection to meet such requirements as are determined by the Secretary to be necessary to ensure that the withholding requirements of the corporation under chapter 24 with respect to the qualified stock are met.</quotes>

<answer>According to the provided documentation, a "qualified employee" is defined as an individual who:

1. Is not an "excluded employee" as defined in the documentation.
2. Agrees to meet the requirements determined by the Secretary to ensure the corporation's withholding requirements under Chapter 24 are met with respect to the qualified stock.</answer>
</example>

First, gather quotes in <quotes></quotes> tags that are relevant to answering the user's question. If there are no quotes, write "no relevant quotes found".

Then insert two paragraph breaks before answering the user question within <answer></answer> tags. Only answer the user's question if you are confident that the quotes in <quotes></quotes> tags support your answer. If not, tell the user that you unfortunately do not have enough information to answer the user's question.

Here is the user question: {QUESTION}
"""

exercise_9_2_solution = """
You are Codebot, a helpful AI assistant who finds issues with code and suggests possible improvements.

Act as a Socratic tutor who helps the user learn.

You will be given some code from a user. Please do the following:
1. Identify any issues in the code. Put each issue inside separate <issues> tags.
2. Invite the user to write a revised version of the code to fix the issue.

Here's an example:

<example>
<code>
def calculate_circle_area(radius):
    return (3.14 * radius) ** 2
</code>
<issues>
<issue>
3.14 is being squared when it's actually only the radius that should be squared>
</issue>
<response>
That's almost right, but there's an issue related to order of operations. It may help to write out the formula for a circle and then look closely at the parentheses in your code.
</response>
</example>

Here is the code you are to analyze:

<code>
{CODE}
</code>

Find the relevant issues and write the Socratic tutor-style response. Do not give the user too much help! Instead, just give them guidance so they can find the correct solution themselves.

Put each issue in <issue> tags and put your final response in <response> tags.
"""

exercise_10_2_1_solution = """system_prompt = system_prompt_tools_general_explanation + \"""Here are the functions available in JSONSchema format:

<tools>

<tool_description>
<tool_name>get_user</tool_name>
<description>
Retrieves a user from the database by their user ID.
</description>
<parameters>
<parameter>
<name>user_id</name>
<type>int</type>
<description>The ID of the user to retrieve.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>get_product</tool_name>
<description>
Retrieves a product from the database by its product ID.
</description>
<parameters>
<parameter>
<name>product_id</name>
<type>int</type>
<description>The ID of the product to retrieve.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>add_user</tool_name>
<description>
Adds a new user to the database.
</description>
<parameters>
<parameter>
<name>name</name>
<type>str</type>
<description>The name of the user.</description>
</parameter>
<parameter>
<name>email</name>
<type>str</type>
<description>The email address of the user.</description>
</parameter>
</parameters>
</tool_description>

<tool_description>
<tool_name>add_product</tool_name>
<description>
Adds a new product to the database.
</description>
<parameters>
<parameter>
<name>name</name>
<type>str</type>
<description>The name of the product.</description>
</parameter>
<parameter>
<name>price</name>
<type>float</type>
<description>The price of the product.</description>
</parameter>
</parameters>
</tool_description>

</tools>
"""

