# What am I trying to do here?
# - I want to be able to extract the lines of therapy from some document context (in this case its a progress note)

# The reason i want to extract this data is so i can do the following types of queries
# - how many patients have at least one line of met therapy
# - group patients by primary diagnosis and lines of met therapy
# - how many patients have at least one line of met therapy and have at any point been treated with folfox


# to do this you need the following functionality

# provide some document context as text
# provide a system prompt
# provide the model to use
# provide the output model you want to use for the extraction
# provide a validator that will validate the output
# provide the max number of iterations you want to allow the model to run

# what will then happen is this data will be provided to an agent that can do the following
# - do the initial extraction
# - run the validation loop where it
# -- validates the output is correctly in line with the original system prompt
# -- provides feedback if the output is not correct
# -- uses that feedback to create search and replace tool calls to modify the ouptut
# -- does this in a loop until max iterations hit or the output is complete

# NOTE: because we're doing search and replace we might have to provide the output of the model back in as json and then allow the model to do a search and replace on that, then convert it back into the structured output model i originally wanted
# - so you need the ability to convert structured output to json which you can do with model dump json, then edit that json, then convert that json back into the original structured mode - shouldn't be too hard since you're using pydantic

# so how do you know this worked ?
# - well for us right now we'll be doing a unit test where we simply mock the output of the llm so that we can just make sure this setup is working
# - then we'll do an integration test where we run it to make sure it works with live data, once that's working you can start doing the evals i suppose

# now that i'm saying this though i think the smart move might actually be to start with the evals, cause there all you need is
# - document context
# - system prompt
# - model to use
# - validator
# - max iterations

# - expected output
# -- so given some input, what are you expecting the output to be, and this can be llm as judge which will likely be the move, but you can also validate some things deterministicly, for example number of lines of therapy you were expecting, number in metestatic setting, etc. - the thing you might need llm as judge for are the actual regimens and perhaps the ordering of them or something.

# keep in mind that you need to get this info out so that you can setup the infrastructure to enable you to ultimately provide document context, system prompt, and output model, and have that data extracted.

# once you can extract that data, then you can save it to a db and have that data queried so you can start doing some really good feasability analysis.

# so what's the simplest thing you can do right now
# - you need an output model for the resulting extraction for lines of therapy


# at a high level what i need is
# - regimen name
# - specific drugs
# - drug class


# a line of therapy can have multiple components for combo, so i feel like a line of therapy should be
# - regimen name
# - specific drugs each drug should have its own:
# -- drug class
# -- admin route
# - disease setting - non-met, locally-advanced, met
# - treamtment intent
# - start date
# - optional end date (optional because it might not be finished yet)
# - current status
# - reason for change
# - supporting evidence - list of strings (excerpts from document context)
# - confidence score
# - notes - optional


# the extraction then is just:
# - list of lines of therapy
# - optional list of extraction challenges (should be short and brief)


# TODO - need a deidentified progress note to use for this once you get it runing

# then you need to clean this up, there's a lot of shared stuff that you'll be doing, for example the whole validation loop and search and replace and the model being used is all going to be shared across this type of thing, so we can refactor later but just wanted to remind myself to do it


from pydantic import BaseModel, Field, field_validator, model_validator
from typing import Optional, List
from datetime import date, datetime
from enum import Enum
from dataclasses import dataclass, field as dataclass_field, asdict
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from config.settings import settings
import pytest
import json


class DiseaseSettingEnum(str, Enum):
    """Classification of disease extent"""

    NON_METASTATIC = "non-metastatic"
    LOCALLY_ADVANCED = "locally-advanced"
    METASTATIC = "metastatic"
    UNKNOWN = "unknown"


class TreatmentIntentEnum(str, Enum):
    """Intent of therapy"""

    NEOADJUVANT = "neoadjuvant"
    ADJUVANT = "adjuvant"
    PALLIATIVE = "palliative"
    CURATIVE = "curative"
    UNKNOWN = "unknown"


class DrugClassEnum(str, Enum):
    """Type of anti-cancer drug"""

    CHEMOTHERAPY = "chemotherapy"
    TARGETED_THERAPY = "targeted_therapy"
    IMMUNOTHERAPY = "immunotherapy"
    HORMONAL_THERAPY = "hormonal_therapy"
    RADIOPHARMACEUTICAL = "radiopharmaceutical"
    UNKNOWN = "unknown"


class AdministrationRouteEnum(str, Enum):
    """Route of administration"""

    ORAL = "oral"
    INTRAVENOUS = "intravenous"
    SUBCUTANEOUS = "subcutaneous"
    INTRAMUSCULAR = "intramuscular"
    UNKNOWN = "unknown"


class CurrentStatusEnum(str, Enum):
    """Status of this line"""

    ONGOING = "ongoing"
    COMPLETED = "completed"
    DISCONTINUED = "discontinued"
    UNKNOWN = "unknown"


class Drug(BaseModel):
    """Individual drug within a regimen"""

    name: str = Field(description="Drug name (generic preferred)", min_length=1)

    drug_class: DrugClassEnum = Field(description="Type of anti-cancer drug")

    administration_route: AdministrationRouteEnum = Field(
        description="How this drug is administered"
    )


class LineOfTherapy(BaseModel):
    """Single line of systemic anti-cancer therapy"""

    regimen_name: str = Field(
        description="Name of the regimen (e.g., 'FOLFOX', 'palbociclib + letrozole')",
        min_length=1,
    )

    specific_drugs: List[Drug] = Field(
        description="Individual drugs in this regimen", min_length=1
    )

    disease_setting: DiseaseSettingEnum = Field(
        description="Disease extent at time of treatment"
    )

    treatment_intent: TreatmentIntentEnum = Field(description="Intent of the therapy")

    start_date: date = Field(description="When treatment started")

    end_date: Optional[date] = Field(
        None, description="When treatment ended (null if ongoing)"
    )

    current_status: CurrentStatusEnum = Field(description="Current status of this line")

    reason_for_change: Optional[str] = Field(
        None, description="Why therapy was stopped or changed"
    )

    supporting_evidence: List[str] = Field(
        default_factory=list, description="Relevant excerpts from progress notes"
    )

    confidence_score: float = Field(
        description="Confidence in extraction (0-1)", ge=0.0, le=1.0
    )

    notes: Optional[str] = Field(None, description="Additional clarifications")

    @field_validator("end_date")
    @classmethod
    def validate_end_date(cls, v, info):
        """Ensure end_date is not before start_date"""
        if v is not None and "start_date" in info.data:
            if v < info.data["start_date"]:
                raise ValueError("end_date cannot be before start_date")
        return v

    @model_validator(mode="after")
    def validate_status_consistency(self):
        """Ensure status matches end_date"""
        if (
            self.current_status == CurrentStatusEnum.ONGOING
            and self.end_date is not None
        ):
            raise ValueError("Cannot be ongoing with an end_date")

        if self.current_status in [
            CurrentStatusEnum.COMPLETED,
            CurrentStatusEnum.DISCONTINUED,
        ]:
            if self.end_date is None:
                raise ValueError(
                    f"Status {self.current_status.value} requires an end_date"
                )

        return self


class LinesOfTherapyExtraction(BaseModel):
    """Complete extraction of lines of therapy"""

    lines_of_therapy: List[LineOfTherapy] = Field(
        description="All lines of therapy in chronological order"
    )

    extraction_challenges: Optional[List[str]] = Field(
        None, description="Brief notes on any extraction difficulties"
    )

    @field_validator("lines_of_therapy")
    @classmethod
    def validate_chronological(cls, v):
        """Ensure chronological order"""
        if len(v) < 2:
            return v

        for i in range(len(v) - 1):
            if v[i].start_date > v[i + 1].start_date:
                raise ValueError(f"Line {i+1} starts after Line {i+2}")

        return v


# ============================================================================
# ORCHESTRATION MODELS FOR MARKDOWN-BASED EXTRACTION
# ============================================================================


class MarkdownOutput(BaseModel):
    """Structured output for markdown generation"""

    markdown: str = Field(
        description="The generated markdown representation of lines of therapy"
    )


class ValidationResult(BaseModel):
    """Structured output for markdown validation"""

    is_valid: bool = Field(description="Whether the markdown meets all requirements")
    feedback: Optional[str] = Field(
        default=None,
        description="Detailed feedback on what needs to be fixed (only if invalid)",
    )
    specific_issues: List[str] = Field(
        default_factory=list, description="List of specific issues found"
    )


class SearchReplaceOperation(BaseModel):
    """Single search and replace operation"""

    old_string: str = Field(description="The exact text to find and replace")
    new_string: str = Field(description="The replacement text")
    reason: str = Field(description="Why this replacement is needed")


class ToolCallPlan(BaseModel):
    """Structured output for tool call generation"""

    operations: List[SearchReplaceOperation] = Field(
        description="List of search/replace operations to perform"
    )
    rationale: str = Field(description="Overall rationale for these changes")


@dataclass
class ValidationAttempt:
    """Single validation attempt with feedback"""

    iteration: int
    markdown_content: str
    is_valid: bool
    feedback: Optional[str] = None
    timestamp: datetime = dataclass_field(default_factory=datetime.now)


@dataclass
class ToolCallApplication:
    """Single tool call (search/replace) application"""

    iteration: int
    old_string: str
    new_string: str
    success: bool
    timestamp: datetime = dataclass_field(default_factory=datetime.now)


@dataclass
class ValidationState:
    """Comprehensive state tracking for the validation process"""

    # Core data
    initial_progress_note: str
    initial_markdown: str
    current_markdown: str

    # Process tracking
    current_iteration: int = 0
    max_iterations: int = 3
    is_complete: bool = False
    final_result: Optional[str] = None

    # History tracking
    validation_history: List[ValidationAttempt] = dataclass_field(default_factory=list)
    tool_call_history: List[ToolCallApplication] = dataclass_field(default_factory=list)

    def add_validation_attempt(self, is_valid: bool, feedback: Optional[str] = None):
        """Record a validation attempt"""
        attempt = ValidationAttempt(
            iteration=self.current_iteration,
            markdown_content=self.current_markdown,
            is_valid=is_valid,
            feedback=feedback,
        )
        self.validation_history.append(attempt)

        if is_valid:
            self.is_complete = True
            self.final_result = self.current_markdown

    def add_tool_call(self, old_string: str, new_string: str, success: bool):
        """Record a tool call application"""
        tool_call = ToolCallApplication(
            iteration=self.current_iteration,
            old_string=old_string,
            new_string=new_string,
            success=success,
        )
        self.tool_call_history.append(tool_call)

    def get_context_summary(self) -> str:
        """Generate a summary of current state for agents"""
        summary = f"""
        VALIDATION CONTEXT:
        - Current Iteration: {self.current_iteration}/{self.max_iterations}
        - Total Validation Attempts: {len(self.validation_history)}
        - Total Tool Calls Applied: {len(self.tool_call_history)}
        - Current Status: {'Complete' if self.is_complete else 'In Progress'}

        RECENT VALIDATION FEEDBACK:
        """

        # Include last 3 validation attempts
        for attempt in self.validation_history[-3:]:
            summary += f"\n  Iteration {attempt.iteration}: {'VALID' if attempt.is_valid else 'INVALID'}"
            if attempt.feedback:
                summary += f"\n    Feedback: {attempt.feedback}"

        # Include recent tool calls
        if self.tool_call_history:
            summary += f"\n\nRECENT TOOL CALLS:"
            for tool_call in self.tool_call_history[-3:]:
                summary += f"\n  Iteration {tool_call.iteration}: {'SUCCESS' if tool_call.success else 'FAILED'}"
                summary += f"\n    Changed: '{tool_call.old_string[:50]}...' -> '{tool_call.new_string[:50]}...'"

        return summary.strip()


class ExtractionResult(BaseModel):
    """Final result from the extraction process"""

    success: bool = Field(description="Whether the process completed successfully")
    extraction: Optional[LinesOfTherapyExtraction] = Field(
        default=None, description="The final extracted lines of therapy"
    )
    iterations_used: int = Field(description="Number of iterations required")
    total_issues_found: int = Field(description="Total number of issues encountered")
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )


# ============================================================================
# SYSTEM PROMPTS
# ============================================================================

EXTRACTION_SYSTEM_PROMPT = """You are an expert medical data extraction specialist. Your task is to extract lines of systemic anti-cancer therapy from medical progress notes and represent them in a structured markdown format.

## What Counts as a Line of Therapy

Include all systemic anti-cancer therapies:
- IV/oral chemotherapy (FOLFOX, carboplatin/paclitaxel, capecitabine, etc.)
- Targeted therapies any route (osimertinib, palbociclib, trastuzumab, bevacizumab)
- Immunotherapy (pembrolizumab, nivolumab, ipilimumab)
- Hormonal therapies (tamoxifen, letrozole, abiraterone, enzalutamide)
- Systemic radiopharmaceuticals (Lu-177 PSMA, I-131, Ra-223, Y-90)
- Concurrent chemoradiation (count the chemo component as one line)

Do NOT include:
- Standalone radiation (EBRT, SBRT, brain/bone radiation)
- Surgery
- Local therapies only (RFA, cryotherapy)
- Supportive care (antiemetics, growth factors, bisphosphonates for bone health)

## When to Start a NEW Line

- Disease progression and treatment changed
- New disease setting (e.g., completed adjuvant therapy → years later developed metastases → started new therapy)
- Unplanned drug change due to progression, resistance, or severe intolerance
- Different regimen started after completing prior course

## When to CONTINUE the Same Line

- Planned sequential therapies (AC x4 then paclitaxel x4 = one line)
- Cycle progression (C1 → C2 → C3 of same regimen)
- Dose modifications (reduced osimertinib 80mg to 40mg)
- Brief treatment holds that resume (held for toxicity then restarted)
- Schedule adjustments (pembro Q3 weeks → Q6 weeks)
- Planned maintenance that was part of original treatment plan

## Treatment Pattern Recognition

Traditional IV chemotherapy: Progress notes say "Cycle 4 of FOLFOX" or "C3D1". Extract start date from first cycle, end date when explicitly stopped or completed.

Oral targeted therapies: Progress notes say "continues on osimertinib" or "tolerating erlotinib well". These run continuously until progression. Start when first mentioned, end only when explicitly stopped.

Immunotherapy: Progress notes say "pembrolizumab Q3 weeks" or "received dose #8". Treat like continuous therapy - regular dosing but not discrete "cycles". Start when first dose given, end when stopped.

Combination regimens: List all drugs. Each drug has its own drug_class and administration_route.

Concurrent chemoradiation: Extract the systemic chemotherapy as one line. Document that it was given with radiation in the notes field. Disease_setting usually locally-advanced, treatment_intent usually curative.

## Key Principles

Dates: Use YYYY-MM-DD format. If only month/year given, use first of month (June 2023 = 2023-06-01). Calculate relative dates from note date. Always provide best estimate.

End dates: Must be null if therapy is currently ongoing. Only provide when therapy explicitly stopped or changed.

Disease setting:
- non-metastatic: Early stage, no distant mets
- locally-advanced: Unresectable local disease, no distant mets
- metastatic: Stage IV, distant metastases present
- unknown: Can't determine

Treatment intent:
- neoadjuvant: Before surgery to shrink tumor
- adjuvant: After surgery to prevent recurrence
- curative: Intent to cure (some locally-advanced cases)
- palliative: For metastatic disease to extend life
- unknown: Can't determine

Drug classification:
- chemotherapy: Traditional cytotoxics
- targeted_therapy: Drugs targeting specific mutations/pathways
- immunotherapy: Checkpoint inhibitors
- hormonal_therapy: Endocrine agents
- radiopharmaceutical: Radioactive therapeutic agents
- unknown: Can't determine

Supporting evidence: Include relevant excerpts from progress notes showing treatment start, changes, progression, and end.

Confidence score:
- 0.9-1.0 = all info clear and exact
- 0.7-0.9 = most info clear, minor inference needed
- 0.5-0.7 = moderate ambiguity
- 0.3-0.5 = significant uncertainty
- <0.3 = very uncertain

## Common Mistakes to Avoid

- Don't count palliative radiation to bone/brain as a line
- Don't break up planned sequential therapies into separate lines
- Don't create new line for dose adjustments of same regimen
- Don't confuse brief treatment holds with treatment endings
- Don't count medication refills as new lines

## Output Format

Generate a markdown document with the following structure for each line of therapy:

```
## Line of Therapy 1
Regimen: [regimen name]
Disease Setting: [non-metastatic|locally-advanced|metastatic|unknown]
Treatment Intent: [neoadjuvant|adjuvant|curative|palliative|unknown]
Start Date: YYYY-MM-DD
End Date: YYYY-MM-DD or null
Status: [ongoing|completed|discontinued|unknown]
Reason for Change: [reason or null]

Drugs:
  - Name: [drug name] | Class: [chemotherapy|targeted_therapy|immunotherapy|hormonal_therapy|radiopharmaceutical|unknown] | Route: [oral|intravenous|subcutaneous|intramuscular|unknown]
  - Name: [drug name] | Class: [...] | Route: [...]

Supporting Evidence:
  - "[excerpt from note]"
  - "[excerpt from note]"

Confidence: [0.0-1.0]
Notes: [optional notes]

---
```

Separate each line of therapy with `---` on its own line.
"""

VALIDATION_SYSTEM_PROMPT = """You are a meticulous validator checking extracted lines of therapy data. Your job is to verify that the markdown representation accurately captures all relevant information from the original progress note.

## Validation Checks

1. **Completeness**: Are all lines of therapy mentioned in the progress note captured in the markdown?
2. **Accuracy**: Do the dates, drug names, disease settings, and intents match the source material?
3. **Proper Line Determination**: Are therapies correctly grouped or separated as lines?
4. **Format Compliance**: Does the markdown follow the expected structure?
5. **Supporting Evidence**: Are there relevant excerpts that justify the extraction?
6. **Confidence Appropriateness**: Does the confidence score reflect the clarity of information?

## What to Flag

- Missing lines of therapy
- Incorrect dates or drug names
- Wrong disease setting or treatment intent classifications
- Therapies that should be combined but are separated (or vice versa)
- Missing or weak supporting evidence
- Formatting issues

Provide specific, actionable feedback on what needs to be corrected."""

CORRECTION_SYSTEM_PROMPT = """You are a precise editor. Given validation feedback on extracted lines of therapy markdown, generate search-and-replace operations to fix the identified issues.

## Instructions

1. Read the validation feedback carefully
2. For each issue, create a SearchReplaceOperation that:
   - Provides the exact old_string to find (must be unique and present in current markdown)
   - Provides the corrected new_string
   - Explains the reason for the change

## Guidelines

- Be precise - old_string must match exactly
- Include enough context in old_string to make it unique
- Fix one issue at a time
- Keep changes minimal and targeted
- Preserve markdown formatting"""


# ============================================================================
# AGENT CREATION FUNCTIONS
# ============================================================================


def get_default_model():
    """Get default model configuration (defaults to OpenAI)"""
    return get_openai_model()


def get_anthropic_model():
    """Get Anthropic Claude model configuration"""
    if not settings.anthropic_api_key:
        raise ValueError("ANTHROPIC_API_KEY not configured in settings")

    model_settings = AnthropicModelSettings(
        max_tokens=25000,
    )
    return AnthropicModel(
        "claude-sonnet-4-5",
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
        settings=model_settings,
    )


def get_openai_model():
    """Get OpenAI model configuration"""
    if not settings.openai_api_key:
        raise ValueError("OPENAI_API_KEY not configured in settings")

    model_settings = OpenAIResponsesModelSettings(max_tokens=25000)
    return OpenAIResponsesModel(
        "gpt-5",
        provider=OpenAIProvider(api_key=settings.openai_api_key),
        settings=model_settings,
    )


def create_extraction_agent(model) -> Agent:
    """Create extraction agent for generating markdown from progress notes"""
    return Agent(
        model=model,
        output_type=MarkdownOutput,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
    )


def create_validation_agent(model) -> Agent:
    """Create validation agent for checking markdown accuracy"""
    validator = Agent(
        model=model,
        output_type=ValidationResult,
        system_prompt=VALIDATION_SYSTEM_PROMPT,
        deps_type=ValidationState,
    )

    @validator.instructions
    async def add_validation_context(ctx: RunContext[ValidationState]) -> str:
        """Add dynamic context about the validation history"""
        context = f"""
        VALIDATION CONTEXT:
        {ctx.deps.get_context_summary()}

        ORIGINAL PROGRESS NOTE LENGTH: {len(ctx.deps.initial_progress_note)} characters
        CURRENT MARKDOWN LENGTH: {len(ctx.deps.current_markdown)} characters

        INITIAL MARKDOWN:
        {ctx.deps.initial_markdown}

        CURRENT MARKDOWN:
        {ctx.deps.current_markdown}
        """
        return context

    return validator


def create_correction_agent(model) -> Agent:
    """Create correction agent for generating search/replace operations"""
    correction_agent = Agent(
        model=model,
        output_type=ToolCallPlan,
        system_prompt=CORRECTION_SYSTEM_PROMPT,
        deps_type=ValidationState,
    )

    @correction_agent.instructions
    async def add_correction_context(ctx: RunContext[ValidationState]) -> str:
        """Add context for tool call generation"""
        return f"""
        CURRENT ITERATION: {ctx.deps.current_iteration}

        CURRENT MARKDOWN TO FIX:
        {ctx.deps.current_markdown}

        ORIGINAL PROGRESS NOTE:
        {ctx.deps.initial_progress_note}
        """

    return correction_agent


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def apply_corrections(
    markdown: str, operations: List[SearchReplaceOperation]
) -> tuple[str, int]:
    """
    Apply search and replace operations to markdown

    Returns:
        Tuple of (modified_markdown, successful_operations_count)
    """
    successful_operations = 0
    current_markdown = markdown

    for op in operations:
        if op.old_string in current_markdown:
            print(f"  ✓ Applying: {op.reason}")
            current_markdown = current_markdown.replace(op.old_string, op.new_string, 1)
            successful_operations += 1
        else:
            print(f"  ✗ Could not find text to replace: {repr(op.old_string[:100])}")

    return current_markdown, successful_operations


async def extract_to_pydantic(markdown: str, model) -> LinesOfTherapyExtraction:
    """Convert validated markdown to Pydantic model"""
    extraction_agent = Agent(
        model=model,
        output_type=LinesOfTherapyExtraction,
        system_prompt="You are a precise data parser. Convert the provided markdown representation of lines of therapy into the structured LinesOfTherapyExtraction model. Preserve all information accurately.",
    )

    prompt = f"""
    Convert the following markdown representation of lines of therapy into the LinesOfTherapyExtraction model:

    {markdown}
    """

    print("\n🔄 Converting markdown to Pydantic model...")
    result = await extraction_agent.run(prompt)
    return result.output


# ============================================================================
# MAIN ORCHESTRATION FUNCTION
# ============================================================================


async def extract_lines_of_therapy_async(
    progress_note: str, max_iterations: int = 3
) -> ExtractionResult:
    """
    Extract lines of therapy from a progress note using iterative validation

    Args:
        progress_note: Raw progress note text
        max_iterations: Maximum validation/correction iterations (default 3)

    Returns:
        ExtractionResult with extracted data or error

    Raises:
        ValueError: If validation fails after max_iterations
    """
    model = get_default_model()

    print("=" * 80)
    print("🚀 Starting Lines of Therapy Extraction")
    print("=" * 80)

    # Step 1: Generate initial markdown
    print("\n📝 Step 1: Generating initial markdown from progress note...")
    print("-" * 60)

    extraction_agent = create_extraction_agent(model)
    result = await extraction_agent.run(
        f"Extract lines of therapy from this progress note:\n\n{progress_note}"
    )
    initial_markdown = result.output.markdown.strip()

    # Clean up any markdown formatting
    if "```markdown" in initial_markdown:
        initial_markdown = (
            initial_markdown.split("```markdown")[1].split("```")[0].strip()
        )
    elif "```" in initial_markdown:
        initial_markdown = initial_markdown.split("```")[1].strip()

    print(f"\n✅ Generated markdown ({len(initial_markdown.splitlines())} lines)")
    print("\n" + initial_markdown)
    print("-" * 60)

    # Step 2: Initialize validation state
    state = ValidationState(
        initial_progress_note=progress_note,
        initial_markdown=initial_markdown,
        current_markdown=initial_markdown,
        max_iterations=max_iterations,
    )

    print(f"\n🔄 Step 2: Starting validation loop (max {max_iterations} iterations)")
    print("-" * 60)

    validation_agent = create_validation_agent(model)
    correction_agent = create_correction_agent(model)

    # Main validation loop
    for iteration in range(max_iterations):
        state.current_iteration = iteration
        print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Validate current markdown
        validation_prompt = f"""
        Validate the following markdown extraction against the original progress note.
        
        ORIGINAL PROGRESS NOTE:
        {state.initial_progress_note}
        
        CURRENT MARKDOWN:
        {state.current_markdown}
        """

        print("🔍 Validating markdown...")
        validation_result = await validation_agent.run(validation_prompt, deps=state)
        validation_output = validation_result.output

        # Record validation attempt
        state.add_validation_attempt(
            is_valid=validation_output.is_valid, feedback=validation_output.feedback
        )

        if validation_output.is_valid:
            print("✅ Validation passed!")
            print(validation_output.model_dump_json(indent=2))
            break

        # Show validation feedback
        print("❌ Validation failed:")
        print(validation_output.model_dump_json(indent=2))

        # Generate corrections
        correction_prompt = f"""
        The markdown validation failed with this feedback:
        {validation_output.feedback}
        
        Specific issues:
        {chr(10).join(f"- {issue}" for issue in validation_output.specific_issues)}
        
        Generate search-and-replace operations to fix these issues.
        """

        print("\n🔧 Generating corrections...")
        correction_result = await correction_agent.run(correction_prompt, deps=state)
        operations = correction_result.output.operations

        print(f"📝 Generated {len(operations)} correction operations")
        print(correction_result.output.model_dump_json(indent=2))

        # Apply corrections
        print("\n⚙️  Applying corrections...")
        state.current_markdown, successful_ops = apply_corrections(
            state.current_markdown, operations
        )

        # Record tool calls
        for op in operations:
            success = op.old_string in initial_markdown or successful_ops > 0
            state.add_tool_call(op.old_string, op.new_string, success)

        print(f"✅ Applied {successful_ops}/{len(operations)} operations")

        print(f"\n📄 Markdown after iteration {iteration + 1}:")
        print("-" * 40)
        print(state.current_markdown)
        print("-" * 40)

    # Check if validation succeeded
    if not state.is_complete:
        error_msg = f"Validation failed after {max_iterations} iterations"
        print(f"\n❌ {error_msg}")
        print("\nValidation history:")
        print(
            json.dumps(
                [asdict(attempt) for attempt in state.validation_history],
                indent=2,
                default=str,
            )
        )

        raise ValueError(
            f"{error_msg}. Last feedback: {state.validation_history[-1].feedback if state.validation_history else 'None'}"
        )

    # Step 3: Final extraction to Pydantic model
    print("\n📦 Step 3: Converting to Pydantic model...")
    print("-" * 60)

    extraction = await extract_to_pydantic(state.current_markdown, model)

    print(
        f"✅ Successfully extracted {len(extraction.lines_of_therapy)} lines of therapy"
    )

    result = ExtractionResult(
        success=True,
        extraction=extraction,
        iterations_used=state.current_iteration + 1,
        total_issues_found=len([v for v in state.validation_history if not v.is_valid]),
        error_message=None,
    )

    print("\n" + "=" * 80)
    print("🎉 Extraction Complete!")
    print(result.model_dump_json(indent=2))
    print("=" * 80)

    return result


# ============================================================================
# SAMPLE DATA
# ============================================================================

SAMPLE_PROGRESS_NOTE = """
PATIENT: John Doe
MRN: 12345678
DATE OF NOTE: 2024-08-15

DIAGNOSIS: Metastatic colorectal cancer

HISTORY OF PRESENT ILLNESS:
Mr. Doe is a 62-year-old male with a history of stage IV colorectal cancer diagnosed in January 2023. 
He initially presented with abdominal pain and was found to have a sigmoid colon mass with liver metastases.

TREATMENT HISTORY:

The patient was started on first-line FOLFOX chemotherapy on 01/15/2023 for his metastatic disease. 
He received a total of 8 cycles (C1-C8). Treatment was generally well-tolerated with manageable neuropathy. 
Restaging scans in June 2023 showed disease progression with new lung metastases. FOLFOX was discontinued 
on 06/20/2023 due to progression.

Given progression on FOLFOX, we transitioned to second-line therapy with FOLFIRI plus bevacizumab starting 
07/10/2023. He is currently on cycle 6 of this regimen (C6D1 today). Recent scans from last month show 
stable disease. He continues to tolerate treatment well with mild fatigue and occasional diarrhea managed 
with supportive care.

CURRENT MEDICATIONS:
- Irinotecan 180 mg/m2 IV every 2 weeks
- 5-fluorouracil 400 mg/m2 IV bolus, then 2400 mg/m2 continuous infusion over 46 hours every 2 weeks
- Leucovorin 400 mg/m2 IV every 2 weeks
- Bevacizumab 5 mg/kg IV every 2 weeks

ASSESSMENT AND PLAN:
Continue current FOLFIRI + bevacizumab regimen. Patient is tolerating well with stable disease. 
Will continue current plan and reassess with imaging in 2 months.

Next appointment in 2 weeks for C7D1.
"""


# ============================================================================
# TEST FUNCTIONS
# ============================================================================


async def test_extract_lines_of_therapy():
    """Integration test for lines of therapy extraction with real API"""
    # Run extraction
    result = await extract_lines_of_therapy_async(
        SAMPLE_PROGRESS_NOTE, max_iterations=3
    )

    # Verify result
    assert result.success is True
    assert result.extraction is not None
    assert len(result.extraction.lines_of_therapy) == 2

    # Check first line (FOLFOX)
    line1 = result.extraction.lines_of_therapy[0]
    assert "FOLFOX" in line1.regimen_name
    assert line1.disease_setting == DiseaseSettingEnum.METASTATIC
    assert line1.treatment_intent == TreatmentIntentEnum.PALLIATIVE
    assert line1.current_status == CurrentStatusEnum.COMPLETED
    assert line1.end_date is not None
    assert len(line1.specific_drugs) >= 2  # Should have at least 5-FU and oxaliplatin

    # Check second line (FOLFIRI + bevacizumab)
    line2 = result.extraction.lines_of_therapy[1]
    assert "FOLFIRI" in line2.regimen_name or "bevacizumab" in line2.regimen_name
    assert line2.disease_setting == DiseaseSettingEnum.METASTATIC
    assert line2.treatment_intent == TreatmentIntentEnum.PALLIATIVE
    assert line2.current_status == CurrentStatusEnum.ONGOING
    assert line2.end_date is None  # Should be ongoing
    assert len(line2.specific_drugs) >= 3  # Should have irinotecan, 5-FU, bevacizumab

    print("\n" + "=" * 80)
    print("TEST PASSED!")
    print("=" * 80)
    print(f"\nExtracted {len(result.extraction.lines_of_therapy)} lines of therapy:")
    print(result.extraction.model_dump_json(indent=2))
