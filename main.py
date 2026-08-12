import textwrap
from langgraph.types import interrupt
import subprocess
from typing import Annotated
from typing import TypedDict
from langgraph.graph import END,START, StateGraph
from langgraph.types import Send
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI
import base64
from dotenv import load_dotenv
import operator
load_dotenv()


memory = InMemorySaver()


llm = init_chat_model(model="openai:gpt-4o-mini")

class State(TypedDict):
    video_file:str
    audio_file:str
    transcription:str
    summaries: Annotated[list[str],operator.add]
    thumbnail_prompts: Annotated[list[str],operator.add]
    thumbnail_ideas: Annotated[list[str],operator.add]
    final_summary:str
    user_feedback:str
    chosen_prompt:int


def extract_audio(state:State):
    output_file = state["video_file"].replace("mp4","mp3")
    command = [
        "ffmpeg",
        "-i",
        state["video_file"],
        "-filter:a",
        "atempo=2.0",
        "-y",
        output_file
    ]
    subprocess.run(command)
    return{
        "audio_file":output_file

    }

def transcribe_audio(state:State):
    client = OpenAI()
    with open(state["audio_file"],"rb") as audio_file:  
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            response_format="text",
            file=audio_file,
            language="en",
            prompt="Spiderman, Peter Parker, Dr Banner,"
        )
        return {
            "transcription": transcription
        }


def dispatch_summarizers(state:State):
    transcripts = state["transcription"]
    chunks = []
    for i, chunk in enumerate(textwrap.wrap(transcripts,500)):
        chunks.append({"id":i+1,"chunk":chunk})
    return [Send("summarize_chunk",chunk) for chunk in chunks]

def summarize_chunk(chunk):
    chunk_id = chunk["id"]
    chunk = chunk["chunk"]
    response = llm.invoke(
        f"""
        Summarize the following text.
        Text: {chunk}
         """
    )
    summary = f"[Chunk {chunk_id}] {response.content}"
    return{
        "summaries":[summary]
    }


def final_summary(state:State):
    all_summaries = "\n".join(state["summaries"])
    prompt = f"""
        You are given multiple summaries of different chunks from a video transcription.
        Create a comprehensive final summary that combines all the key points.
        Individual summaries :
        {all_summaries}
     """
    response = llm.invoke(prompt)
    return {
        "final_summary":response.content
     }

def dispatch_idea_artists(state:State):
    return [Send("generate_idea_thumbnail",{"id":i, "summary":state["final_summary"]}) for i in [1,2,3]]

def generate_idea_thumbnail(args):
    id = args["id"]
    summary = args["summary"]
    prompt = f"""  
    Based on this video summary, create a detailed visual prompt for a Youtube thumbnail.
    Create a detailed prompt for generating a thumbnail image that would attract the viewers. Try to use
    very specifics and real authentic relevant stuff only by searching for it and grounding it in both real world context and provided summary. Include:
     - Main Visual Elements
     - Color scheme (Mood)
     - Text overlay suggestion
     - Overall professional,clean and sleek tone
     - Overall composition

     CRITICAL INSTRUCTION: Do NOT include copyrighted character names (like Spider-Man, Peter Parker, etc.) or franchise names in the generated prompt. Instead, describe them generically.

     Summary:{summary}
    """
    response = llm.invoke(prompt)

    thumbnail_prompt = response.content

    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto"
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)
    filename = f"thumbnail_{id}.jpg"
    with open(filename,"wb") as file:
        file.write(image_bytes)
    return {
        "thumbnail_prompts":[thumbnail_prompt],
        "thumbnail_ideas":[filename]
    }

    response = llm.invoke(prompt)

    thumbnail_prompt = response.content

    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=thumbnail_prompt,
        quality="low",
        moderation="low",
        size="auto"
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)
    filename = f"thumbnail_{id}.jpg"
    with open(filename,"wb") as file:
        file.write(image_bytes)
    return {
        "thumbnail_prompts":[thumbnail_prompt],
        "thumbnail_ideas":[filename]
    }

def human_feedback(state:State):
    answer = interrupt({
        "chosen_thumbnail":"Which thumbnail among these do you like?",
        "feedback":"Provide any feedback , direction or changes you'd like for the final thumbnail."
    })
    user_feedback = answer["user_feedback"]
    chosen_prompt = answer["chosen_prompt"]
    return{
        "user_feedback":user_feedback,
        "chosen_prompt":state["thumbnail_prompts"][chosen_prompt-1]
    }


def gen_hd_thumbnail(state:State):
    user_feedback = state["user_feedback"]
    chosen_prompt = state["chosen_prompt"]
    final_hd_thumbnail_prompt = f"""
        You are an expert AI image-prompt engineer. Your job is to write ONE final, production-ready image generation prompt for the FINAL YouTube thumbnail — this is the last revision, so the prompt you write must be flawless and complete on the first try. You are NOT generating the image yourself; you are writing the exact prompt text that will be sent to an image generation model.

        === BASE CONCEPT (approved by user) ===
        {chosen_prompt}

        === USER REVISION NOTES (must be applied exactly) ===
        {user_feedback}

        Treat the revision notes as override instructions: wherever they conflict with the base concept, the revision notes win. Do not silently ignore, soften, or reinterpret any note — apply each one literally.

        Write the final prompt so that it satisfies every rule below. Weave these into natural, cohesive prompt prose — do not output them as a checklist or with these section headers; the headers below are instructions for you, not text to include in your output.

        TECHNICAL SPECIFICATION
        - High-contrast, thumb-stopping composition, legible even at small mobile feed size
        - Style: photorealistic / high-production digital art (match whatever style is implied by the base concept) — NOT flat vector, NOT generic stock-photo look, NOT airbrushed "AI face"

        COMPOSITION RULES
        1. ONE clear focal point. The eye should land on the subject within 0.5 seconds — no competing elements of equal visual weight.
        2. Subject occupies 40-65% of frame, positioned using rule-of-thirds (not dead-center unless the concept specifically calls for symmetry).
        3. Foreground subject in sharp focus; background may be blurred/darkened (depth of field) to push subject forward.
        4. Lighting: directional, high-contrast key light on the subject's face/focal object. Avoid flat, shadowless lighting.
        5. Color: 2-3 dominant colors max, saturated and high-contrast against each other, so the thumbnail reads instantly against a white/dark YouTube UI.
        6. Leave clear negative space (top-left or bottom-right third) for title text overlay if the concept includes text — do not fill that zone with detail.

        FACE & HUMAN ANATOMY (if a person is present)
        - Exactly ONE consistent face per person shown — no duplicated, merged, or asymmetrical facial features
        - Eyes: both fully open, same size, correctly aligned, natural catchlight
        - Hands (if visible): exactly 5 fingers per hand, correct joints — if uncertain, specify the shot cropped so hands are out of frame
        - Expression matches the emotional tone in the base concept (shock, excitement, curiosity, etc.) — exaggerated but not uncanny

        TEXT RENDERING (if any text/overlay is specified)
        - Any on-image text must be spelled EXACTLY as written in the base concept or revision notes — character for character, no substitutions, no invented words
        - Font: bold, thick-stroke, high legibility at small sizes
        - Text must have a contrasting outline, drop shadow, or solid background plate so it never blends into the background
        - Do NOT add any text, watermark, label, or logo that was not explicitly requested

        HARD CONSTRAINTS — the prompt you write must exclude these
        - No extra limbs, extra fingers, floating objects, or physically impossible geometry
        - No blurry, low-resolution, or compressed-looking output
        - No duplicated subjects unless explicitly requested
        - CRITICAL: NO unintended brand logos, copyrighted characters (e.g., Spider-Man, Peter Parker, Marvel), or real public figures AT ALL. If they are named in the base concept or revision notes, you MUST replace them with a generic visual description (e.g., 'a young superhero in a red and blue webbed suit').
        - No border, frame, or watermark added around the image
        - Background must be fully rendered, no empty/unfinished regions

        OUTPUT FORMAT
        Output ONLY the final image generation prompt itself, as one cohesive block of prompt text ready to send directly to an image model. Do not include any preamble, explanation, headers, numbering, or meta-commentary like "Here is the prompt:". Do not restate these instructions.
"""
    response = llm.invoke(final_hd_thumbnail_prompt)
    final_thumbnail_prompt = response.content

    client = OpenAI()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=final_thumbnail_prompt,
        quality="high",
        moderation="low",
        size="auto"
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)
    filename = "thumbnail_final.jpg"
    with open(filename,"wb") as file:
        file.write(image_bytes)


graph_builder = StateGraph(State)
graph_builder.add_node("extract_audio",extract_audio)
graph_builder.add_node("transcribe_audio",transcribe_audio)
graph_builder.add_node("summarize_chunk",summarize_chunk)
graph_builder.add_node("final_summary",final_summary)
graph_builder.add_node("generate_idea_thumbnail",generate_idea_thumbnail)
graph_builder.add_node("human_feedback",human_feedback)
graph_builder.add_node("gen_hd_thumbnail",gen_hd_thumbnail)

graph_builder.add_edge(START,"extract_audio")
graph_builder.add_edge("extract_audio","transcribe_audio")
graph_builder.add_conditional_edges("transcribe_audio",dispatch_summarizers,["summarize_chunk"])
graph_builder.add_edge("summarize_chunk","final_summary")
graph_builder.add_conditional_edges("final_summary",dispatch_idea_artists,["generate_idea_thumbnail"])
graph_builder.add_edge("generate_idea_thumbnail","human_feedback")
graph_builder.add_edge("human_feedback","gen_hd_thumbnail")
graph_builder.add_edge("gen_hd_thumbnail",END)



graph = graph_builder.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id":"1"
}}


result = graph.invoke({"video_file":"spiderman-bnd.mp4"},config=config)