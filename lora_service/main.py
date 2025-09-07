import os
from typing import TypedDict, Annotated, List
from fastapi import FastAPI, Body
from pydantic import BaseModel
import json

# FastAPI app
app = FastAPI(title="LoRA Service")

class InputData(BaseModel):
    descriptions: List[str]

class OutputItem(BaseModel):
    description_id: int
    image_links: List[str]

class OutputData(BaseModel):
    results: List[OutputItem]

@app.post("/train", response_model=OutputData)
async def process_descriptions(data: OutputData = Body(...)):
    print(f"LORA SERVICE: {data}")
    return "--%LORA ANSWERRRRR%--"

