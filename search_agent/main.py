import os
from typing import TypedDict, Annotated, List
from fastapi import FastAPI, Body
from pydantic import BaseModel
from agent import graph
import json
import requests

# FastAPI app
app = FastAPI(title="Agent Service")

class InputData(BaseModel):
    descriptions: List[str]

class OutputItem(BaseModel):
    description_id: int
    image_links: List[str]

class OutputData(BaseModel):
    results: List[OutputItem]

@app.post("/process_descriptions", response_model=OutputData)
async def process_descriptions(data: InputData = Body(...)):
    results = []
    for i, desc in enumerate(data.descriptions):
        inputs = {"input_text": desc}
        result = graph.invoke(inputs)
        links = result.get("final_output", [])
        results.append({"description_id": i, "image_links": links})

    try:
        print(f'\n----{results}----')

        lora_response = requests.post(
            "http://krakend:8080/lora/train",
            json=results,
            timeout=30
        )
        lora_response.raise_for_status()
        lora_data = lora_response.json()
        return {"results": results,"lora": lora_data}
        
        # for lora_item in lora_data.get("results", []):
        #     desc_id = lora_item["description_id"]
        #     for result in results:
        #         if result["description_id"] == desc_id:
        #             result["generated_image_urls"] = lora_item.get("generated_image_urls", [])
        #             break
    except requests.RequestException as e:
        print(f"Error calling lora_service: {e}")

