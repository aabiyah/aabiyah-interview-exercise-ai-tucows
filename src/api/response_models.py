# Using Pydantic to define request and response models.
from pydantic import BaseModel, Field
from typing import List, Optional


class TicketRequest(BaseModel):
# Request model for customer support ticket.
    ticket_text: str = Field(
        ...,
        description="Customer support ticket text",
        min_length=10,
        max_length=2000,
        example="My domain was suspended and I didn't get any notice. How can I reactivate it?"
    )


class TicketResponse(BaseModel):
    # Response model for resolved customer support ticket.
    answer: str = Field(
        ...,
        description="Generated answer to the customer query"
    )
    references: List[str] = Field(
        default_factory=list,
        description="List of FAQ sources cited"
    )
    action_required: str = Field(
        ...,
        description="Required action (none|escalate_to_abuse_team|needs_human_review|contact_provider)"
    )
    confidence_score: Optional[float] = Field(
        None,
        description="Confidence score (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    reasoning_trace: Optional[str] = Field(
        None,
        description="Internal reasoning (debug mode only)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Your domain may have been suspended due to a policy violation or missing WHOIS information. Please update your WHOIS details and contact support.",
                "references": ["FAQ: Domain Suspension Guidelines"],
                "action_required": "escalate_to_abuse_team",
                "confidence_score": 0.82
            }
        }
