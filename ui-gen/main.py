from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv
from fasthtml.common import *

load_dotenv()

client = OpenAI()
app, rt = fast_app()


@app.get("/")
def home():
    headers = (
        Script(src="https://cdn.tailwindcss.com"),
        Link(
            rel="stylesheet",
            href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css",
        ),
    )

    content = Title("GenUI Demo"), Main(
        *headers,
        Div(
            Div(
                Div(
                    H1("Gen UI", cls="text-4xl font-bold text-gray-800"),
                    P(
                        "Nothing too fancy, but still kind of fancy.",
                        cls="text-lg text-gray-600",
                    ),
                    cls="text-center mb-8",
                ),
            ),
            Div(
                Form(
                    Textarea(
                        type="text",
                        name="prompt",
                        id="prompt",
                        placeholder="Enter your prompt",
                        cls="textarea textarea-bordered w-full",
                    ),
                    Button("Generate UI", type="submit", cls="btn btn-primary mt-2"),
                    id="prompt-form",
                    hx_trigger="submit",
                    hx_post="/genui",
                    hx_target="#custom-ui",
                    hx_swap="innerHTML",
                    cls="mb-8",
                ),
            ),
            Div(
                id="custom-ui",
            ),
            cls="container mx-auto p-4 max-w-3xl",
        ),
        cls="min-h-screen bg-base-200",
    )
    # return Html(*headers, d)
    return content


@app.post("/genui")
def generate_ui(data: dict):
    print("generate UI")
    print(data)

    class HTMLComponent(BaseModel):
        plan: str = Field(
            ...,
            description="Let's think step by step how we gonna implement this, what are key component & structure, which css class to use",
        )
        html: str

    class AIResponse(BaseModel):
        components: List[HTMLComponent]

    prompt = data["prompt"] if data["prompt"] else "Generate some random UI"
    system_message = """
    You are an AI assistant that generates HTML components using DaisyUI classes.
    The user will provide instructions, and you should respond with appropriate HTML components.
    Only use DaisyUI classes for styling. Do not use any custom CSS or other CSS frameworks.
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        response_format=AIResponse,
    )

    print(completion)

    ui = completion.choices[0].message.parsed
    print(ui)
    html_output = ui.components[0].html
    print(html_output)

    return f"""<div class="card bg-base-100 shadow-xl"><div class="card-body">Generated UI <div>{html_output}</div></div></div>"""


serve()