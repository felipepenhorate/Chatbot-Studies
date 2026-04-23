# Factory-Based DSPy Wrapper for LangChain Runnables
# This module provides an alternative wrapper that creates fresh LangGraph
# instances for each invocation, completely eliminating thread-safety issues.
# Use this when the standard wrapper still has issues with GEPA or other
# parallel optimizers.

from typing import Any, Callable, List, Type, Union

import dspy
from dspy import Signature
from langchain_core.runnables import Runnable
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage


class LangChainAgentModuleFactory(dspy.Module):
    """DSPy Module that wraps a LangChain runnable/agent using a factory pattern.

    Instead of reusing the same LangGraph instance, this creates a NEW instance
    for each invocation. This completely eliminates thread-safety issues that
    arise during parallel optimization.

    Args:
        signature: DSPy signature defining inputs/outputs and system
            prompt (via docstring).
        agent_factory: Function that creates a NEW LangGraph agent
            when called. Must take no arguments and return
            a compiled LangGraph (a LangChain Runnable).
        system_prompt_field: Name of the field to store the optimizable
            system prompt (default: "system_prompt").
    """

    def __init__(
        self,
        signature: Union[str, Type[Signature]],
        agent_factory: Callable[[], Runnable],
        system_prompt_field: str = "system_prompt",
    ):
        super().__init__()
        self.signature = signature
        self.agent_factory = agent_factory
        self.system_prompt_field = system_prompt_field

        # Create a Predict module with the signature prompt
        self.predictor = dspy.Predict(signature)

        # Initial system prompt comes from the signature docstring, if any
        if hasattr(signature, "__doc__") and signature.__doc__:
            self.initial_system_prompt = signature.__doc__.strip()
        else:
            self.initial_system_prompt = ""

    def _extract_system_prompt(self) -> str:
        """Extract the current system prompt from the predictor."""
        if hasattr(self.predictor, "extended_signature"):
            sig = self.predictor.extended_signature
        else:
            sig = self.predictor.signature

        if hasattr(sig, "instructions") and sig.instructions:
            return sig.instructions.strip()
        elif hasattr(sig, "__doc__") and sig.__doc__:
            return sig.__doc__.strip()

        return self.initial_system_prompt

    def _prepare_langchain_messages(
        self,
        system_prompt: str,
        **kwargs: Any,
    ) -> List[BaseMessage]:
        """Prepare messages for LangChain agent."""
        messages: List[BaseMessage] = []
        messages.append(SystemMessage(content=system_prompt))

        input_text_parts: List[str] = []
        for key, value in kwargs.items():
            if value is not None and value != "":
                input_text_parts.append(f"### {key}: {value}\n\n")

        if input_text_parts:
            user_content = "\n".join(input_text_parts)
            messages.append(HumanMessage(content=user_content))

        return messages

    def _extract_response(self, result: Any) -> str:
        """Extract the text response from LangChain agent result."""
        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            # Common keys where an answer might live
            for key in ["output", "response", "answer", "content", "text"]:
                if key in result:
                    last_msg = result[key]
                    if hasattr(last_msg, "content"):
                        return last_msg.content
                    elif isinstance(last_msg, dict):
                        return last_msg.get("content", str(last_msg))
                    else:
                        return str(last_msg)

            # Sometimes messages are returned
            if "messages" in result:
                msgs = result["messages"]
                if isinstance(msgs, list) and msgs:
                    last = msgs[-1]
                    if hasattr(last, "content"):
                        return last.content
                    return str(last)

        # Fallbacks
        if hasattr(result, "content"):
            return result.content
        if isinstance(result, list) and result:
            last = result[-1]
            if hasattr(last, "content"):
                return last.content
            return str(last)

        return str(result)

    def _get_output_fields(self) -> List[str]:
        """Get output field names from signature."""
        # String signatures like "input -> output"
        if isinstance(self.signature, str):
            output_part = self.signature.split("->")[-1].strip()
            return [output_part]

        # Structured signatures with output_fields metadata
        sig_instance = self.signature
        output_fields: List[str] = []

        if hasattr(sig_instance, "output_fields"):
            for field_name, field_info in sig_instance.output_fields.items():
                field_type = getattr(field_info, "json_schema_extra", {}).get(
                    "_dspy_field_type"
                )
                if field_type == "output":
                    output_fields.append(field_name)

        return output_fields if output_fields else ["response"]

    def get_system_prompt(self) -> str:
        """Get the current system prompt."""
        return self._extract_system_prompt()

    def set_system_prompt(self, prompt: str) -> None:
        """Manually set the system prompt."""
        if hasattr(self.predictor, "extended_signature"):
            self.predictor.extended_signature.instructions = prompt
        self.initial_system_prompt = prompt

    def forward(self, **kwargs: Any) -> dspy.Prediction:
        """
        Execute the LangChain agent with the current system prompt.

        Creates a FRESH agent instance for each call, eliminating all
        thread-safety issues.

        Args:
            **kwargs: Input fields defined in the signature.

        Returns:
            dspy.Prediction with the agent's response.
        """
        # Get the current (possibly optimized) system prompt
        system_prompt = self._extract_system_prompt()

        # Create a FRESH agent instance for this invocation
        # This eliminates ALL thread-safety issues!
        agent = self.agent_factory()

        # Prepare input
        messages = self._prepare_langchain_messages(system_prompt, **kwargs)

        try:
            # Invoke the fresh agent
            result = agent.invoke({"messages": messages})

            # Validate result is a dict (as expected from LangGraph)
            if not isinstance(result, dict):
                raise TypeError(
                    f"LangGraph agent must return a dict, got {type(result)}"
                )

            # Extract response from result
            response = self._extract_response(result)

            # Get output field name
            output_field_names = self._get_output_fields()
            output_field_name = (
                output_field_names[0] if output_field_names else "response"
            )

            # CRITICAL: Call predictor to register in DSPy’s trace
            # but use agent’s response instead of an LLM call
            original_forward = self.predictor.forward
            agent_response_value = response

            def mock_forward(**pred_kwargs: Any) -> dspy.Prediction:
                # Return agent’s response without calling LLM
                return dspy.Prediction({output_field_name: agent_response_value})

            try:
                # Replace forward temporarily
                self.predictor.forward = mock_forward

                # Call predictor so DSPy can trace this module
                pred = self.predictor(**kwargs)
                return pred
            finally:
                # Restore original forward
                self.predictor.forward = original_forward

        except Exception as e:
            error_msg = (
                f"LangChain agent invocation failed! {str(e)}\n"
                f"Input kwargs: {list(kwargs.keys())}"
            )
            raise RuntimeError(error_msg) from e
