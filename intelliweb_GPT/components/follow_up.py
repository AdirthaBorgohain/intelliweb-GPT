import os
from annotated_types import Len
from typing import List, Dict, Annotated
from json.decoder import JSONDecodeError
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, wait_fixed, stop_after_attempt, retry_if_exception_type

from intelliweb_GPT.llms import load_llm
from intelliweb_GPT.prompts import FOLLOW_UP_QUERY_CREATION

from llama_index.core.llms import LLM
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.output_parsers import PydanticOutputParser


class FollowUpQueries(BaseModel):
    """
    Pydantic Class that aids in generating follow-up queries. It consists of a single field - 'queries'.
    'queries' is a List of 3 queries which could be used as follow-up queries considering the given user
    query & model response
    """
    queries: Annotated[List[str], Len(min_length=3, max_length=3)] = Field(
        description="List of follow-up queries based on the given dialogue."
    )


class FollowUpQueryCreator:
    _REFERENCES_MARKER = '--- \n\n  \nReferences '

    def _extract_content_from_message_dict(self, message: str):
        # Find the start and end indexes for content extraction
        start_index = 0
        end_index = message.find(self._REFERENCES_MARKER) if self._REFERENCES_MARKER in message else len(message)
        # Extracting the text between 'Answer Started' and 'References'
        content = message[start_index:end_index].strip()

        # Return the extracted content and references.
        return content or message

    def _serialize_conversation(self, conversation: List[Dict]) -> str:
        # Define a dictionary mapping roles in the conversation to their human-readable forms
        role_mappings = {"user": "User", "assistant": "AI"}
        # Initialize a list to hold serialized messages
        messages = []

        for message_dict in conversation:
            # Extract content from the current message's content
            content = self._extract_content_from_message_dict(message_dict['content'])
            # Append the formatted message to the messages list. If the message is from the assistant, add a separator
            messages.append(
                f"{role_mappings.get(message_dict['role'])}: {content}" +
                ("\n\n" + "-" * 60 + "\n" if message_dict['role'] == 'assistant' else "")
            )
        # Concatenate all messages into a single string and eliminate duplicate references, returning both
        return "\n".join(messages) + "\n"

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(0.2),
           retry=(retry_if_exception_type((JSONDecodeError, ValidationError))))
    async def create_follow_up_queries(self, conversation: List[Dict], model: None | LLM = None) -> List[str]:
        llm = model or load_llm(model=os.getenv('GPT_MODEL_LITE', 'gpt-4o'))

        program = LLMTextCompletionProgram.from_defaults(
            output_parser=PydanticOutputParser(FollowUpQueries),
            prompt_template_str=FOLLOW_UP_QUERY_CREATION,
            llm=llm,
            verbose=True,
        )

        serialized_conversation = self._serialize_conversation(conversation)

        try:
            output = await program.acall(serialized_conversation=serialized_conversation)
            print("Generating follow-up queries")
            return output.queries
        except:
            print("Failed to generate any follow-up queries.")
        return []


__all__ = ['FollowUpQueryCreator']

if __name__ == "__main__":
    import asyncio

    query_creator = FollowUpQueryCreator()
    response = asyncio.run(query_creator.create_follow_up_queries(
        conversation=
        [
            {
                "role": "user",
                "content": "Could you begin by explaining the pathophysiological roles of LRP4 and MUSK in the neuromuscular junction and how they are implicated in Myasthenia Gravis?"},
            {
                "role": "assistant",
                "content": "**Chain of Thought and Plans:** \n\n --- \n\n**icon-mdi-brain[d-inline-block ms-1 akr-text-orange] Thought Process:** Understanding the roles of LRP4 and MUSK in the neuromuscular junction requires access to detailed scientific explanations and research findings. Scholarly articles or comprehensive reviews in neurology or immunology journals are likely to provide in-depth information on how these proteins function and their significance in neuromuscular transmission, as well as their implication in diseases like Myasthenia Gravis.\n\n**icon-mdi-lightbulb[d-inline-block ms-1 akr-text-orange] Plan:** Search for scholarly articles or reviews that detail the pathophysiological roles of LRP4 and MUSK in the neuromuscular junction. **Source:** Web\n\n --- \n\n**icon-mdi-brain[d-inline-block ms-1 akr-text-orange] Thought Process:** Recent case studies or clinical research can provide insights into how abnormalities or mutations in LRP4 and MUSK are clinically associated with Myasthenia Gravis. This approach will help understand the real-world implications of these proteins' dysfunction in patients, offering a more applied perspective on the pathophysiological mechanisms discussed in theoretical research.\n\n**icon-mdi-lightbulb[d-inline-block ms-1 akr-text-orange] Plan:** Find recent case studies or clinical research focusing on patients with Myasthenia Gravis, specifically looking at abnormalities or mutations in LRP4 and MUSK. **Source:** Web\n\n --- \n\n**icon-mdi-brain[d-inline-block ms-1 akr-text-orange] Thought Process:** A large language model can provide a concise summary of the roles of LRP4 and MUSK in the neuromuscular junction and their implications in Myasthenia Gravis. This is useful for obtaining a quick overview or a simplified explanation that integrates various aspects of scientific research and clinical observations. It's an effective way to complement the detailed information obtained from scholarly articles and clinical research.\n\n**icon-mdi-lightbulb[d-inline-block ms-1 akr-text-orange] Plan:** Use a large language model to summarize the roles of LRP4 and MUSK in the neuromuscular junction and their implications in Myasthenia Gravis, based on existing knowledge. **Source:** LLM\n\n --- \n\n\n\n **Response to the Question:** \n\n<!-- Processing Compiling -->\n\n<!-- Answer Started -->\n\nThe neuromuscular junction (NMJ) is a critical site for nerve-muscle communication, essential for voluntary muscle contraction and movement. The integrity and proper function of the NMJ depend on the orchestrated interaction of several proteins, among which Lipoprotein Receptor-Related Protein 4 (LRP4) and Muscle-Specific Kinase (MuSK) play pivotal roles. Their pathophysiological roles in the NMJ and implications in Myasthenia Gravis (MG) are complex and multifaceted.\n\n### LRP4 and MuSK in the Neuromuscular Junction\n\nLRP4 and MuSK are integral components of the NMJ, with distinct yet complementary roles:\n\n- **LRP4** acts as a receptor for agrin, a motor neuron-derived protein. Upon binding to agrin, LRP4 undergoes a conformational change that facilitates its interaction with MuSK.\n- **MuSK** is a receptor tyrosine kinase that, upon activation by LRP4-agrin interaction, initiates several downstream signaling pathways. These pathways are crucial for the clustering of acetylcholine receptors (AChRs) at the postsynaptic membrane, a process essential for efficient synaptic transmission.\n\nThe interaction between LRP4 and MuSK, therefore, is critical for the structural and functional organization of the NMJ, ensuring that nerve impulses can be effectively transmitted to muscle fibers.\n\n### Implication in Myasthenia Gravis\n\nMyasthenia Gravis (MG) is an autoimmune disorder characterized by muscle weakness and fatigue. This condition arises from the immune system's production of autoantibodies that target components of the NMJ, disrupting synaptic transmission. In the context of LRP4 and MuSK, the implications are as follows:\n\n- **Anti-LRP4 Antibodies**: These antibodies interfere with the binding of agrin to LRP4, hindering the LRP4-MuSK interaction. This disruption prevents the proper clustering of AChRs, leading to diminished synaptic transmission and muscle weakness.\n- **Anti-MuSK Antibodies**: Targeting MuSK, these antibodies impair the kinase's ability to initiate downstream signaling required for AChR clustering. Patients with anti-MuSK antibodies often exhibit symptoms that rapidly progress, affecting facial and bulbar muscles and potentially leading to severe respiratory issues.\n\nThe presence of these antibodies highlights the heterogeneity of MG and underscores the critical roles of LRP4 and MuSK in maintaining NMJ integrity.\n\n### Mermaid Diagram: LRP4 and MuSK Pathway and Implications in MG\n\n```mermaid\ngraph TB\nA[Agrin] -->|Binds to| B[LRP4]\nB -->|Activates| C[MuSK]\nC -->|Initiates Signaling| D[AChR Clustering]\nD --> E[NMJ Function]\nE -->|Impaired by Anti-LRP4/MuSK Antibodies| F[Myasthenia Gravis Symptoms]\n\nclassDef normal fill:#f9f,stroke:#333,stroke-width:2px;\nclassDef impaired fill:#fbb,stroke:#333,stroke-width:2px;\nclass A,B,C,D,E normal;\nclass F impaired;\n```\n\nThis diagram illustrates the sequential interactions between agrin, LRP4, and MuSK, leading to AChR clustering and NMJ function, and how disruptions caused by anti-LRP4 or anti-MuSK antibodies can lead to MG symptoms.\n\n### Conclusion\n\nThe roles of LRP4 and MuSK at the NMJ are fundamental for muscle function, and their disruption by autoimmune processes underlies the pathogenesis of Myasthenia Gravis. Understanding these mechanisms not only sheds light on the disease's complexity but also opens avenues for targeted therapeutic interventions."
            }
        ]))
    print(response)
