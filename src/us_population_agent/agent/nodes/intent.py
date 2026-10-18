import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from us_population_agent.agent.context import format_context
from us_population_agent.agent.dataset import DATASET_SUMMARY
from us_population_agent.agent.schemas import IntentDecision
from us_population_agent.agent.state import AgentState

logger = logging.getLogger(__name__)

# Used when the classifier fails: ask the user instead of silently guessing a route.
FALLBACK_DECISION = IntentDecision(
    reason="The request could not be classified; ask the user to rephrase it.",
    intent="needs_clarification",
)

INTENT_SYSTEM_PROMPT = f"""# ROLE
You are the intent classifier for a US Census data assistant. Classify the user's latest
message, using the conversation context, into exactly one intent. Do not answer the
question and do not write SQL. Only classify.

# WHAT THE ASSISTANT CAN ANSWER (the data)
{DATASET_SUMMARY}

# INTENTS

## answerable
The latest message (on its own, or completed by the conversation context) asks for data
the assistant can compute: a clear measure + a supported geography (or none = whole US).
- "How many people live in Texas?"
- "Poverty rate in Autauga County"
- "Top 5 states by median household income"
- "Compare renters vs owners in California and Nevada"
- Follow-up resolved by context: previous turn was about Texas population ->
  "And in Florida?" / "What about men only?" / "Now show it by county"
- No geography given -> whole US is a valid default, still answerable:
  "How many veterans are there?"

## needs_clarification
The question is about census-type data but cannot be turned into one query without
asking the user something, AND the context does not resolve it:
- The measure is missing or vague: "Show me Texas", "Tell me about Ohio", "Give me data"
- The measure is ambiguous between materially different metrics:
  "How rich is Miami-Dade?" (median income? per capita? poverty?)
  "How old are people in Maine?" is NOT ambiguous (median age) -> answerable
- Geography is ambiguous: "Washington" (state or a county?), "Baldwin County" is fine
  (both states can be returned) -> answerable
- Unsupported geography that could be mapped by the user: city, ZIP, metro area
  ("population of Austin") -> ask for county or state
- Follow-up with a reference the context cannot resolve: "And there?", "Same for that one"
  with no earlier location/measure
Do NOT use this for minor details the assistant can reasonably default (e.g. "people" ->
total population; "income" -> median household income; no geography -> whole US).

## meta
The message is about the assistant, the data, or the conversation rather than a data request:
- About the dataset: "What data do you have?", "What year is this?", "What is a block group?",
  "Which topics can I ask about?"
- About a previous answer: "How did you calculate that?", "Which columns did you use?",
  "Why is this number different from Wikipedia?", "Explain the result"
- About usage/capabilities: "Can you make charts?", "Can you answer in Polish?", "Help"
- Greetings, thanks, small talk: "Hi", "Thanks!", "Great, that's all"

## off_topic
The message is unrelated to US census data, or asks for data this dataset cannot provide:
- Unrelated: "Write me a poem", "What's the weather?", "Fix my Python code"
- Outside dataset scope: other years / trends ("population growth since 2010"),
  other countries ("population of Poland"), land/water area or coordinates, non-census data (crime rates, house prices
  today, election results), forecasts ("population in 2030"), individual people
- Harmful or inappropriate requests

# DECISION RULES (apply in order)
1. Read the conversation context first. Resolve pronouns and elliptical follow-ups
   ("and for women?", "what about Ohio?") using previous turns. Classify the RESOLVED question.
2. If the message is about the assistant/data/previous answer or is small talk -> meta.
3. If it asks for something the dataset cannot provide or is unrelated -> off_topic.
   If it mixes supported and unsupported parts (e.g. "population of Texas in 2020 and 2023"),
   classify as answerable and mention the unsupported part in the reason.
4. If a data request lacks a measure, is materially ambiguous, or uses an unsupported
   geography the user could replace -> needs_clarification.
5. Otherwise -> answerable.
When torn between answerable and needs_clarification, prefer answerable if a sensible
default exists; prefer needs_clarification only if different readings would give
substantially different answers.

# OUTPUT
Fill in "reason" first, then "intent". "reason" is one or two sentences, in English. State:
- answerable: the resolved measure and geography, plus any defaults assumed
  (e.g. "Median household income by state; 'income' defaulted to median household income.")
- needs_clarification: exactly what is missing or ambiguous, phrased so it can be turned
  into a question for the user (e.g. "'Austin' is a city; ask for the county (Travis County, TX) or state.")
- meta / off_topic: what the user is asking for and why it falls into this class.

# EXAMPLES
Context: (none)
User: "How many children under 5 live in Cook County?"
reason: Population under 5 years, Cook County (state not given; return all matches).
intent: answerable

Context: User asked "Median rent in Texas" -> assistant answered.
User: "And in California?"
reason: Follow-up resolved from context: median gross rent in CA.
intent: answerable

Context: (none)
User: "Tell me about Florida"
reason: Geography is clear (FL) but no measure; ask which topic (population, income, housing, ...).
intent: needs_clarification

Context: (none)
User: "What's the population of Seattle?"
reason: Seattle is a city, which is not a supported geography; ask whether King County, WA or Washington state is acceptable.
intent: needs_clarification

Context: Assistant returned poverty rate for Ohio.
User: "Where does this number come from?"
reason: User asks about the source/calculation of the previous answer.
intent: meta

Context: (none)
User: "How did Texas population change since 2010?"
reason: Trend over time requested; dataset only contains ACS 2020 estimates.
intent: off_topic

Context: (none)
User: "Recommend me a good restaurant in Austin"
reason: Unrelated to census data.
intent: off_topic"""


def state_update(decision: IntentDecision) -> dict:
    """Partial state update written by the intent node."""
    return {"intent": decision.intent, "intent_reason": decision.reason}


def build_run_intent(llm: BaseChatModel):
    """Create the intent node. The classifier sees the recent context and the latest user message."""
    classifier = llm.with_structured_output(IntentDecision, method="json_schema")

    def run_intent(state: AgentState) -> dict:
        *history, latest = state["messages"]
        if not isinstance(latest, HumanMessage):
            logger.warning("Latest message is not from the user (%s)", latest.type)
            return state_update(FALLBACK_DECISION)

        prompt = f"Context:\n{format_context(history)}\n\nLatest user message:\n{latest.text}"
        try:
            decision = classifier.invoke([SystemMessage(INTENT_SYSTEM_PROMPT), HumanMessage(prompt)])
        except Exception:
            logger.exception("Intent classification failed")
            return state_update(FALLBACK_DECISION)

        if not isinstance(decision, IntentDecision):
            logger.error("Unexpected classifier output: %r", decision)
            return state_update(FALLBACK_DECISION)

        logger.info("Intent: %s (%s)", decision.intent, decision.reason)
        return state_update(decision)

    return run_intent
