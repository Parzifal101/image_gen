import os
from typing import TypedDict, Annotated, List
from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_gigachat import GigaChat
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.tools import TavilySearchResults  # Assuming Tavily for web search; replace with your preferred search tool
from langchain_tavily import TavilySearch
from operator import itemgetter

# Set your API keys (replace with actual keys)
os.environ["OPENAI_API_KEY"] = "your_openai_api_key_here"
os.environ["TAVILY_API_KEY"] = "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"  # Or use another search API

# Initialize LLM
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm = ChatMistralAI(mistral_api_key='Qw9IyN7LR7g1iJ2lm0wwVtYWg8N4BAxU',temperature=0)
# llm = GigaChat(credentials="NjkyNjg5MmItNzVkMy00NTM3LWFmNTItYTliMTdkNTE0MDlhOmVkNTQ2YTk0LWM4MDgtNDUwNS05MDc2LWU4YWI1MjdkYjZhNw==", verify_ssl_certs=False, temperature=0,top_p=0.1)
# Define structured output schema
class KeywordOutput(TypedDict):
    keywords: List[str]

# Define tools
@tool
def search_for_images(query: str) -> List[dict]:
    """Поиск изображений в интернете по запросу."""
    search_tool = TavilySearch(max_results=10, include_images=True)
    result = search_tool.invoke({"query": query + " images product photos"})
    print("Search results:", result['images'])
    images = []
    if 'images' in result:
        for img in result.get('images', []):
            if type(img) == str:
                images.append({"url": img, "description": ""})
    elif 'url' in result and 'image' in result.get('content', '').lower():
        images.append({"url": result.get("url"), "description": result.get("content", "")})
    print("Extracted images:", images)
    return images[:5]

@tool
def send_to_microservice(images: List[str], description: str) -> str:
    """Симуляция отправки ссылок на изображения в микросервис для обучения LoRA и генерации фото."""
    print(f"Sending {len(images)} images to microservice: {images}")
    print(f"Description: {description[:100]}...")
    return "Generation complete. Output: simulated_generated_photo_url"

# Bind tools to LLM
llm_with_tools = llm.bind_tools([search_for_images, send_to_microservice])

# Define state
class AgentState(TypedDict):
    input_text: str
    keywords: Annotated[List[str], "Извлеченные ключевые слова для поиска"]
    image_references: Annotated[List[dict], "Найденные ссылки:on images"]
    selected_images: Annotated[List[str], "Выбранные URL изображений для микросервиса"]
    final_output: str

# Node 1: Extract keywords
def extract_keywords(state: AgentState) -> AgentState:
    system_prompt = f"""
        Ты – система обработки описаний товаров для поиска изображений.

        Твоя задача – выделить из текстового описания товара до 10 ключевых фраз, которые подчеркивают визуальные характеристики, бренд и модель продукта. Каждая фраза должна содержать не более пяти слов.

        ## Инструкция
        1. Внимательно проанализируй текстовое описание товара и идентифицируй ключевые термины, относящиеся к бренду, модели и визуальным характеристикам.
        2. Разброс семантики должен быть минимальным. В ИДЕАЛЕ НАЙТИ ТОЧНО ТАКОЙ ЖЕ ТОВАР КОТОРЫЙ В ОПИСАНИИ. Например, если ты понимаешь что в описании есть точное название и модель, то поиск нужно будет сделать сначала по ним
        3. Составляй фразы, содержащие максимум информации при минимальной длине (не больше 5 слов).

        ## Формат ответа
        Верни список из 5–10 фраз, каждую фразу начинай с большой буквы и заканчивай точкой.

        ## Примеры
        *Пример 1*
        Вход: "Современный смартфон Samsung Galaxy S23 Ultra с высоким разрешением экрана и мощным процессором."
        Выход: 
        Samsung Galaxy S23 Ultra
        Высокое разрешение экрана
        Мощный процессор

        *Пример 2*
        Вход: "Красивый кожаный рюкзак Fendi с модным принтом и удобной ручкой."
        Выход: 
        Fendi кожаный рюкзак
        Модный принт
        Удобная ручка

        ## Критерии качества
        - Ключевые слова четко соответствуют описанию бренда, модели и визуальных характеристик.
        - Фразы короткие и емкие, без лишних подробностей.
        - Отсутствуют технические спецификации.



        ВОТ ТЕКСТОВОЕ ОПИСАНИЕ ТОВАРА:{state['input_text']}
    """
    # prompt = ChatPromptTemplate.from_messages([
    #     SystemMessage(content=system_prompt),
    #     MessagesPlaceholder(variable_name="input")
    #     # HumanMessage(content="{input_text}")
    # ])
    # formatted_prompt = prompt.invoke({'input':state['input_text']})
    response = llm.invoke(system_prompt)
    print(response)
    keywords = response.content
    print("Extracted keywords:", keywords)
    return {"keywords": keywords}

# Node 2: Search for references
def search_references(state: AgentState) -> AgentState:
    query = " ".join(state["keywords"]) + " product photos references"
    # Limit query to 400 characters
    query = query[:400]
    print("Search query (truncated to 400 chars):", query)
    images = search_for_images.invoke({"query": query})
    return {"image_references": images}

# Node 3: Select and filter references
def select_references(state: AgentState) -> AgentState:
    selected = [img["url"] for img in state["image_references"] if "url" in img]
    print("SELECTED IMGS:", selected)
    return {"selected_images": selected}

# Node 4: Send to microservice
def process_with_microservice(state: AgentState) -> AgentState:
    # output = send_to_microservice.invoke({
    #     "images": state["selected_images"],
    #     "description": state["input_text"]
    # })
    return {"final_output": state["selected_images"]}

# Define the graph
workflow = StateGraph(AgentState)
workflow.add_node("extract_keywords", extract_keywords)
workflow.add_node("search_references", search_references)
workflow.add_node("select_references", select_references)
workflow.add_node("process_with_microservice", process_with_microservice)

workflow.add_edge("extract_keywords", "search_references")
workflow.add_edge("search_references", "select_references")
workflow.add_edge("select_references", "process_with_microservice")
workflow.add_edge("process_with_microservice", END)
workflow.set_entry_point("extract_keywords")

# Compile the graph
graph = workflow.compile()

# Example usage
if __name__ == "__main__":
    example_description = """
    Общие характеристики
    Артикул производителя  LRc 01182b; LRC01182B
    ОЕМ номер  11190-1300010-40П; 111901301012; FRC1539M; kt104049; HF708430; BTL1118B
    Модель  LRc 01182b
    Дополнительная информация
    Марка автомобиля  Lada
    Страна производства  Россия
    Комплектация  Радиатор охлаждения; Паспорт изделия
    Габариты
    Вес с упаковкой (кг)  3.56 кг
    Длина упаковки  80 см
    Высота упаковки  40 см
    Ширина упаковки  7 см
    Описание
    Радиатор охлаждения LRc 01182b бренда LUZAR - высококачественный и сертифицированный товар, подходящий для следующих марок автомобиля Lada Kalina (04-), KALINA.. Радиатор охлаждения является одной из важнейших запчастей, необходимых для правильной работы двигателя автомобиля. Он отвечает за охлаждение жидкости, циркулирующей в двигателе, и обеспечивает эффективный теплообмен системы охлаждения двигателя. Это очень важная функция, поскольку перегрев двигателя может привести к его поломке. Радиатор для авто Lada Kalina (04-), KALINA. изготовлен из алюминия, который обладает высокой теплопроводностью. Стенки радиатора для автомобиля очень тонкие, благодаря чему антифриз быстро отдает свою температуру. Конструкция радиатора состоит из тонких трубок, спаянных между собой в форму прямоугольника. Этот ель крепится на двух бачках (один на входе, другой на выходе). Дополнительно на трубки нанизаны пластинки, что увеличивает площадь теплоотдачи. Воздух проходит между ребрами и быстро охлаждает поверхность детали. Для проверки применяемости отправьте VIN автомобиля через вкладку Вопросы! Применяемость для А/М:Lada Kalina (04-), Kalina (04-), Kalina (04-), Характеристики: Гарантийный срок: 24 месяца Количество на автомобиль: 1, Трансмиссия: MT, Кондиционер: +, Длина сердцевины, мм: 560, Высота сердцевины, мм: 340,2, Толщина сердцевины, мм: 26, Тип сердцевины: сборная, Аналог: LRc 01183 (паяная сердцевина).
    """
    inputs = {"input_text": example_description}
    result = graph.invoke(inputs)
    print(result["final_output"])