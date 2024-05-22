from dotenv import load_dotenv

load_dotenv()

import asyncio
import chainlit as cl
from intelliweb_GPT import generate_answer
from intelliweb_GPT.components import FollowUpQueryCreator, QueryReframer

query_reframer = QueryReframer()
follow_up_query_creator = FollowUpQueryCreator()


@cl.on_chat_end
async def remove_related_questions_element():
    try:
        msg = cl.user_session.get("related_questions_msg")
        if msg:
            await msg.remove()
            cl.user_session.set("related_questions_msg", None)
    except:
        pass


@cl.action_callback("Thinking and Answering...")
async def on_action(action: cl.Action):
    await remove_related_questions_element()
    msg = cl.Message(content=action.value, type="user_message")
    await msg.send()
    await chat(msg)


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def chat(message: cl.Message):
    """
    This function is called every time a user inputs a message in the UI.
    It sends back an intermediate response from the tool, followed by the final answer.

    Args:
        message: The user's message.

    Returns:
        None.
    """
    await remove_related_questions_element()

    message = message.content

    chat_history = cl.user_session.get("chat_history", [])
    chat_history.append({"role": "user", "content": message})

    chat_params = {
        'stream': True,
        'use_serper_api': True
    }

    reframed_query = query_reframer.reframe_query(chat_history[-5:]) \
        if chat_params.get('use_history') else chat_history[-1]['content']
    chat_params['query'] = reframed_query

    msg = cl.Message(content="")
    await msg.send()

    async with cl.Step(name="intelliwebGPT", type="llm", root=True) as step:
        step.input = message
        response_dict = await generate_answer(**chat_params)
        answer_generator = response_dict['answer_generator']
        references = response_dict['references']

        print("ANSWER GENERATOR: ", answer_generator)
        print("REFERENCES: ", references)
        await asyncio.sleep(5)

        async for token in answer_generator:
            await step.stream_token(token)

        if references:
            references_str = "<ul>"
            for reference in references:
                references_str += "<li><a href={}>{}</a></li>".format(reference, reference)
            references_str += "</ul>"
            await step.stream_token(f"\n<hr><h3>References:</h3>{references_str}")

    await step.update()
    answer = step.output
    chat_history.append({"role": "assistant", "content": answer})
    cl.user_session.set("chat_history", chat_history)

    related_questions = await follow_up_query_creator.create_follow_up_queries(cl.user_session.get("chat_history")[-2:])
    related_questions_actions = [
        cl.Action(name="Thinking and Answering...", label=question, value=question, description=question)
        for question in related_questions
    ]
    msg = cl.Message(content="Related Questions:", actions=related_questions_actions, disable_feedback=True)
    await msg.send()
    cl.user_session.set("related_questions_msg", msg)


@cl.on_chat_start
async def start():
    cl.user_session.set("chat_history", [])
    await cl.Message(
        content="Hi there!\nI am **intelliweb-GPT**, a question answering tool that help you answer your questions by "
                "accessing the internet.\n"
                "<span style='font-size: 12px;'><strong>NOTE</strong>: To improve the quality of your responses, "
                "please ensure that your questions are detailed and clearly articulated, avoiding the reliance solely "
                "on keywords.</span>", disable_feedback=True).send()
