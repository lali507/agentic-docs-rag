"""Gradio chat over the workflow: python -m agentic_rag.app"""
from __future__ import annotations

from typing import Callable

import gradio as gr

from .config import enable_netfree_ssl_if_available, load_settings
from .query_workflow import AnswerResult, QueryWorkflow
from .services import MissingKeyError, get_llm, get_retriever
from .store import KnowledgeStore

EXAMPLES = [
    "איך מתקינים את המערכת?",
    "מה הצבע העיקרי שנבחר לדיזיין של המערכת?",
    "מה ההנחיה העדכנית לגבי שימוש ב-RTL בממשק?",
    "אילו שדות נוספו או הוסרו מטבלאות קיימות?",
    "תן לי רשימה של כל ההחלטות הטכניות שהתקבלו בפרויקט.",
    "האם קיימת מגבלה טכנית שחוזרת בכמה מסמכים שונים?",
]

CSS = """
.gradio-container { direction: rtl; }
.gradio-container h1, .gradio-container p, .gradio-container li { unicode-bidi: plaintext; }
.gradio-container p, .gradio-container li { text-align: start; }
.gradio-container code, .gradio-container pre { direction: ltr; unicode-bidi: isolate; }
"""


def format_result(result: AnswerResult) -> str:
    parts = [result.answer]
    if result.sources:
        listed = "\n".join(f"- `{source}`" for source in result.sources)
        parts.append(f"**מקורות:**\n{listed}")
    if result.trace:
        steps = "\n".join(f"- {line}" for line in result.trace)
        parts.append(f"<details><summary>מסלול העיבוד</summary>\n\n{steps}\n\n</details>")
    return "\n\n".join(parts)


def build_workflow() -> QueryWorkflow:
    settings = load_settings()
    return QueryWorkflow(
        settings=settings,
        llm=get_llm(settings),
        retriever=get_retriever(settings),
        store=KnowledgeStore.load(settings.knowledge_path),
    )


def build_demo(workflow_factory: Callable[[], QueryWorkflow] = build_workflow) -> gr.Blocks:
    # One workflow is enough: each run keeps its own state. A failed build is tried again.
    workflow: QueryWorkflow | None = None

    async def respond(message: str, history: list) -> str:
        nonlocal workflow
        try:
            if workflow is None:
                workflow = workflow_factory()
            result = await workflow.run(question=message)
        except MissingKeyError as error:
            return f"⚠️ חסר מפתח: {error}"
        except Exception as error:
            return f"⚠️ אירעה תקלה בעיבוד השאלה ({type(error).__name__}): {error}"
        return format_result(result)

    return gr.ChatInterface(
        fn=respond,
        title="תשאול תיעוד של כלי Agentic Coding",
        description="שאלו על מה שכלי הקידוד תיעדו בפרויקט: החלטות, הנחיות, אזהרות ושינויים.",
        examples=EXAMPLES,
        chatbot=gr.Chatbot(rtl=True, height=520),
        textbox=gr.Textbox(placeholder="כתבו שאלה...", rtl=True, show_label=False, submit_btn=True),
    )


def main() -> None:
    enable_netfree_ssl_if_available()
    build_demo().launch(css=CSS)


if __name__ == "__main__":
    main()
