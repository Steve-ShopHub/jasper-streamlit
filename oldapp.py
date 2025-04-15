# app.py
import streamlit as st
import pandas as pd
import os
import io
import base64 # To encode PDF for iframe
import fitz   # PyMuPDF
import google.generativeai as genai
from google.generativeai import types
import sys
import json
import re
import time
from datetime import datetime
import traceback # Added for better error reporting if needed


# --- 1. SET PAGE CONFIG (MUST BE FIRST st COMMAND) ---
st.set_page_config(layout="wide", page_title="Facility Agreement Analyzer")

# --- 2. Configuration (API Keys, Constants, Schemas, Prompts) ---
API_KEY_ENV_VAR = "GEMINI_API_KEY"
MODEL_NAME = "gemini-2.5-pro-preview-03-25"
MAX_VALIDATION_RETRIES = 1
RETRY_DELAY_SECONDS = 3
# ... (Paste your ai_response_schema_dict here) ...
ai_response_schema_dict = {
  "type": "array",
  "description": "A list containing numbered questions, their categories, answers, clause references, clause wording, and justifications based on the facility agreement analysis.",
  "items": {
    "type": "object",
    "description": "An object representing a single numbered question, its category, the generated answer, and the specific evidence (clause reference and wording) and justification.",
    "properties": {
      "Question Number": {"type": "integer"},
      "Question Category": {"type": "string"},
      "Question": {"type": "string"},
      "Answer": {"type": "string"},
      "Answer Justification": {"type": "string"},
      "Clause Reference": {"type": "string"},
      "Clause Wording": {"type": "string"},
    },
    "required": [
      "Question Number", "Question Category", "Question", "Answer", "Answer Justification",
      "Clause Reference", "Clause Wording"
    ]
  }
}
AI_REQUIRED_KEYS = set(ai_response_schema_dict['items']['required'])
EXCEL_COLUMN_ORDER = [
    "File Name", "Generation Time", "Question Number", "Question Category",
    "Question", "Answer", "Answer Justification", "Clause Reference", "Clause Wording", "Page" # Added Page
]
# ... (Paste your full_prompt_text here) ...
full_prompt_text = """**Task:**
Your primary task is to meticulously analyse the attached facility agreement document. You MUST answer **every single question** listed below for the specified section. Prioritise accuracy and direct evidence from the text.

**Output Requirements:**
Generate a **single JSON array** as output, strictly adhering to the provided schema. Each object within the array MUST correspond to one question from the filtered list below. Ensure all **required** properties (`Question Number`, `Question Category`, `Question`, `Answer`, `Clause Reference`, `Clause Wording`, `Answer Justification`) are present in every object.

*   **Completeness:** Ensure an object is generated for **every question** in the filtered list, even conditional ones.
*   **Question Number:** Populate the 'Question Number' field with the corresponding number from the original list (1 to 78).
*   **Answers:** Populate the 'Answer' field based on your analysis.
    *   For questions with 'Answer Options', your 'Answer' **MUST** contain one of the provided options exactly as written (unless the option indicates free text like 'BLANK', 'Text', 'Date', in which case provide the specific text/date found).
    *   For 'Multiselect' options, list all applicable options separated by commas.
    *   If the document does not contain the information to answer a question, state 'Information Not Found' in the 'Answer' field.
*   **Evidence:**
    *   Populate the 'Clause Reference' field with the specific clause number(s) or section(s) referenced (e.g., 'Clause 14.1(a)', 'Section 5'). If no specific clause provides the answer, state 'N/A'.
    *   Populate the 'Clause Wording' field with the **exact, full text** of the referenced clause(s) or relevant sentence(s). If 'Clause Reference' is 'N/A', this should also be 'N/A'.
*   **Justification:** Populate the 'Answer Justification' field to briefly explain *why* the referenced clause and wording support the given 'Answer', especially if it's not immediately obvious. If the answer is 'Information Not Found' or 'N/A', explain briefly why (e.g., "Condition not met", "Term not defined in document").
*   **Conditional Questions:** For questions with bracketed conditions (e.g., '[If yes...]'), if the condition is **not met** based on the answer to the preceding question *within the current section*, you MUST still include the question object in the JSON array. In such cases, set the 'Answer', 'Clause Reference', 'Clause Wording', and 'Answer Justification' fields to 'N/A - Condition Not Met'. Ensure the 'Question Number' field is still populated correctly.

**Questions to Answer:**

1.  **Question Category:** Agreement Details
    **Question:** What is the date of the agreement you are reviewing?
    **Answer Options:** Date
2.  **Question Category:** Agreement Details
    **Question:** Has a signed version of the agreement you are reviewing been provided?
    **Answer Options:** Yes, No
3.  **Question Category:** Agreement Details
    **Question:** Governing law
    **Answer Options:** Text *(Extract the specific governing law stated in the agreement)*
4.  **Question Category:** Agreement Details
    **Question:** Please detail the clause number of the governing law clause
    **Answer Options:** BLANK
5.  **Question Category:** Eligibility Part 1
    **Question:** Does the agreement contain express provisions relating to sub-participations by Lenders?
    **Answer Options:** Yes, No
6.  **Question Category:** Eligibility Part 1
    **Question:** [If yes to Q5] Is sub-participation defined?
    **Answer Options:** Yes, No
7.  **Question Category:** Eligibility Part 1
    **Question:** [If yes to Q6] Please set out the relevant definition of sub-participation (including clause reference)
    **Answer Options:** BLANK
8.  **Question Category:** Eligibility Part 1
    **Question:** Does the definition of sub-participation capture SRT?
    **Answer Options:** Option to select: No - definition only relates to transfer of voting rights and/or total return swap or derivatives generally, Yes - definition expressly refers to securitisation, SRT, credit default swaps and/or any risk trade or any of the following key terms "risk protection", "credit protection", "risk mitigation", "risk transfer", "synthetic", "credit default swap", "securitisation", "risk or funded", "funded or risk", "credit mitigation", "risk participation"
9.  **Question Category:** Eligibility Part 1
    **Question:** [If yes to Q8] Select what conditions apply to the Lender's right to sub-participate
    **Answer Options:** Multiselect: None, Borrower/Parent/Obligors consent, Borrower/Parent/Obligors consultation, Notice to Borrower/Parent/Obligors, Notice to Agent, Conditions relating to Permitted Lenders, Restricted Lenders and/or Competitors (or any similar concept), Other
10. **Question Category:** Eligibility Part 1
    **Question:** [If conditions relating to Permitted Lenders, Restricted Lenders and/or Competitors (or any similar concept) selected in Q9] Please detail what restrictions apply
    **Answer Options:** BLANK
11. **Question Category:** Eligibility Part 1
    **Question:** [If Other selected in Q9] What other conditions apply?
    **Answer Options:** BLANK
12. **Question Category:** Eligibility Part 1
    **Question:** [If Borrower/Parent/Obligors consent selected in Q9] Select all conditions that apply to Borrower's/Parent's/Obligors' consent
    **Answer Options:** Multiselect: Not unreasonably withheld, Not unreasonably delayed, Deemed consent after a certain number of days, No conditions apply, Other
13. **Question Category:** Eligibility Part 1
    **Question:** [If Other selected in Q12] What other conditions apply?
    **Answer Options:** BLANK
14. **Question Category:** Eligibility Part 1
    **Question:** [If Deemed consent selected in Q12] Number of days for deemed consent
    **Answer Options:** BLANK
15. **Question Category:** Eligibility Part 1
    **Question:** Is this in days or Business Days?
    **Answer Options:** Days, Business Days
16. **Question Category:** Eligibility Part 1
    **Question:** [If Notice to Borrower/Parent/Obligors selected in Q9] How many days notice to the Borrower/Parent/Obligors are required?
    **Answer Options:** BLANK
17. **Question Category:** Eligibility Part 1
    **Question:** Is the notice period in days or Business Days?
    **Answer Options:** Days, Business Days
18. **Question Category:** Eligibility Part 1
    **Question:** Is a form of notice specified?
    **Answer Options:** Yes, No
19. **Question Category:** Eligibility Part 1
    **Question:** [If yes to Q18] Please detail the form of notice required
    **Answer Options:** BLANK
20. **Question Category:** Eligibility Part 1
    **Question:** [If Borrower/Parent/Obligors consultation selected in Q9] Number of days required for consultation
    **Answer Options:** BLANK
21. **Question Category:** Eligibility Part 2
    **Question:** Do any of the following key terms appear anywhere in the agreement, other than in the definition of sub-participation?
    **Answer Options:** Multiselect: "risk protection", "credit protection", "risk mitigation", "risk transfer", "synthetic", "credit default swap", "securitisation", "risk or funded", "funded or risk", "credit mitigation", "risk participation"
22. **Question Category:** Eligibility Part 2
    **Question:** [If yes/terms selected in Q21] Please detail the relevant provision and clause number
    **Answer Options:** BLANK
23. **Question Category:** Eligibility Part 2
    **Question:** [If yes/terms selected in Q21] Are there any conditions which apply to the provisions containing those key terms?
    **Answer Options:** Multiselect: None, Borrower/Parent/Obligors consent, Borrower/Parent/Obligors consultation, Notice to Borrower/Parent/Obligors, Notice to Agent, Conditions relating to Permitted Lenders, Restricted Lenders and/or Competitors (or any similar concept), Other
24. **Question Category:** Eligibility Part 2
    **Question:** [If conditions relating to Permitted Lenders, Restricted Lenders and/or Competitors (or any similar concept) selected in Q23] Please detail what restrictions apply
    **Answer Options:** BLANK
25. **Question Category:** Eligibility Part 2
    **Question:** [If Other selected in Q23] What other conditions apply?
    **Answer Options:** BLANK
26. **Question Category:** Eligibility Part 2
    **Question:** [If Borrower/Parent/Obligors consent selected in Q23] Select all conditions that apply to Borrower's/Parent's/Obligors' consent
    **Answer Options:** Multiselect: Not unreasonably withheld, Not unreasonably delayed, Deemed consent after a certain number of days, No conditions apply, Other
27. **Question Category:** Eligibility Part 2
    **Question:** [If Other selected in Q26] What other conditions apply?
    **Answer Options:** BLANK
28. **Question Category:** Eligibility Part 2
    **Question:** [If Deemed consent selected in Q26] Number of days for deemed consent
    **Answer Options:** BLANK
29. **Question Category:** Eligibility Part 2
    **Question:** Is this in days or Business Days?
    **Answer Options:** Days, Business Days
30. **Question Category:** Eligibility Part 2
    **Question:** [If Notice to Borrower/Parent/Obligors selected in Q23] How many days notice to the Borrower/Parent/Obligors are required?
    **Answer Options:** BLANK
31. **Question Category:** Eligibility Part 2
    **Question:** Is the notice period in days or Business Days?
    **Answer Options:** Days, Business Days
32. **Question Category:** Eligibility Part 2
    **Question:** Is a form of notice specified?
    **Answer Options:** Yes, No
33. **Question Category:** Eligibility Part 2
    **Question:** [If yes to Q32] Please detail the form of notice required
    **Answer Options:** BLANK
34. **Question Category:** Eligibility Part 2
    **Question:** [If Borrower/Parent/Obligors consultation selected in Q23] Number of days required for consultation
    **Answer Options:** BLANK
35. **Question Category:** Eligibility
    **Question:** Based on the questions in section Eligibility Part 1 and section Eligibility Part 2 above, please confirm if the facility you are reviewing can be included in a SRT without restrictions or conditions
    **Answer Options:** Yes, No
36. **Question Category:** Eligibility
    **Question:** [If no to Q35] Please provide further details
    **Answer Options:** BLANK
37. **Question Category:** Confidentiality and Disclosure
    **Question:** Does the agreement contain confidentiality/disclosure provisions?
    **Answer Options:** Yes, No
38. **Question Category:** Confidentiality and Disclosure
    **Question:** [If yes to Q37] Please insert the relevant clause
    **Answer Options:** BLANK
39. **Question Category:** Confidentiality and Disclosure
    **Question:** Is disclosure expressly permitted to any persons entering or potentially entering into transaction under which payments are to be made or may be made by reference to one or more Finance Documents (with or without restriction)?
    **Answer Options:** Yes, No
40. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant provision including clause reference
    **Answer Options:** BLANK
41. **Question Category:** Confidentiality and Disclosure
    **Question:** Is disclosure expressly permitted to any of the following parties in so far as they relate to any persons entering or potentially entering into transaction under which payments are made or may be made by reference to one or more Finance Documents (with or without restriction)?
    **Answer Options:** Multiselect: Affiliates, Professional Advisers, Representatives, Related Funds, All of the above, None of the above
42. **Question Category:** Confidentiality and Disclosure
    **Question:** Is a confidentiality undertaking/other form of NDA required to disclose to any of these parties?
    **Answer Options:** Multiselect: Affiliates, Professional Advisers, Representatives, Related Funds, Persons entering or potentially entering into transaction under which payments are to be made or may be made by reference to one or more Finance Documents, No confidentiality undertaking/other form of NDA required
43. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant definition of "Confidentiality Undertaking"
    **Answer Options:** BLANK
44. **Question Category:** Confidentiality and Disclosure
    **Question:** [If Q42 indicates undertaking required] Is there a prescribed form of confidentiality undertaking / other form of NDA?
    **Answer Options:** Yes, No
45. **Question Category:** Confidentiality and Disclosure
    **Question:** What form of confidentiality undertaking/other form of NDA is specified?
    **Answer Options:** Multiselect: LMA form, Form agreed between the parties to the Facility Agreement, In the form set out in Schedule/Annex/Appendix/Side letter to loan document, LSTA form, Other form
46. **Question Category:** Confidentiality and Disclosure
    **Question:** [If 'In the form set out...' selected in Q45] Is the form of confidentiality undertaking substantially in a recommended form of the LMA?
    **Answer Options:** Yes, No
47. **Question Category:** Confidentiality and Disclosure
    **Question:** [If 'Other form' selected in Q45] Is the form of confidentiality undertaking substantially in a recommended form of the LMA?
    **Answer Options:** Yes, No
48. **Question Category:** Confidentiality and Disclosure
    **Question:** Does the definition of "Confidentiality Undertaking" expressly include any other conditions?
    **Answer Options:** Yes, No
49. **Question Category:** Confidentiality and Disclosure
    **Question:** [If yes to Q48] Please detail any additional conditions which apply
    **Answer Options:** BLANK
50. **Question Category:** Confidentiality and Disclosure
    **Question:** Is Borrower/Obligor/Parent consent required to disclose to any of these parties?
    **Answer Options:** Multiselect: Affiliates, Professional Advisers, Representatives, Related Funds, Persons entering or potentially entering into transaction under which payments are to be made or may be made by reference to one or more Finance Documents, None of the above
51. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant consent provision including clause reference
    **Answer Options:** BLANK
52. **Question Category:** Confidentiality and Disclosure
    **Question:** Is there a requirement for the Borrower, Obligor, Parent to be notified in order for disclosure of confidential information to be permitted to any of these parties?
    **Answer Options:** Multiselect: Affiliates, Professional Advisers, Representatives, Related Funds, Persons entering or potentially entering into transaction under which payments are to be made or may be made by reference to one or more Finance Documents, None of the above
53. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant notification provision including clause reference
    **Answer Options:** BLANK
54. **Question Category:** Confidentiality and Disclosure
    **Question:** If a confidentiality undertaking / other form of NDA is required, is there any requirement to obtain consent from and/ or notify any party in relation to that undertaking/ NDA?
    **Answer Options:** Multiselect: Requirement to obtain consent in relation to the Confidentiality undertaking/NDA, Requirement to notify in relation to the Confidentiality undertaking/NDA, Requirement to send a copy of the Confidentiality Undertaking to the Borrower/Parent/Obligors before disclosure, Requirement to send a copy of the confidentiality undertaking to the Borrower/Parent/Obligors after disclosure, No requirements, No confidentiality undertaking/ other form of NDA required
55. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant consent provision including clause reference.
    **Answer Options:** BLANK
56. **Question Category:** Confidentiality and Disclosure
    **Question:** Is there any restriction in the confidentiality or in the assignment and transfer provisions which prevents the Lender from disclosing information to any of the following?
    **Answer Options:** Multiselect: Permitted Lenders, Restricted Lenders, Competitors (or any similar concept), No restrictions, No Permitted Lender/Restricted Lender/Competitor concepts
57. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant restriction including clause reference.
    **Answer Options:** BLANK
58. **Question Category:** Confidentiality and Disclosure
    **Question:** Ignoring any requirement for a confidentiality undertaking substantially in a form recommended by the LMA, is disclosure expressly permitted to persons [potentially] entering into a transaction under which payments are made/may be made without conditions
    **Answer Options:** Yes, No
59. **Question Category:** Confidentiality and Disclosure
    **Question:** Is disclosure expressly permitted to any Affiliates to National Westminster Bank Plc without restriction or any conditions? Answer 'Yes' even if conditions apply, as long as they only include the accepted exceptions (namely: declaring its confidential nature to whom it is being disclosed, information regarding price sensitivity, recipient's professional obligations to maintain confidentiality).
    **Answer Options:** Yes, No
60. **Question Category:** Confidentiality and Disclosure
    **Question:** Please detail any restrictions or conditions which apply to disclosure to Affiliates. If the conditions found are only part of the accepted exceptions list (namely: declaring its confidential nature to whom it is being disclosed, information regarding price sensitivity, recipient's professional obligations to maintain confidentiality), you do not need to report them; just answer 'No'. Otherwise, detail the conditions.
    **Answer Options:** Text
61. **Question Category:** Confidentiality and Disclosure
    **Question:** Is disclosure expressly permitted to any Representatives to National Westminster Bank Plc without restriction or any conditions? Answer 'Yes' even if conditions apply, as long as they only include the accepted exceptions list (namely: declaring its confidential nature to whom it is being disclosed, information regarding price sensitivity, recipient's professional obligations to maintain confidentiality).
    **Answer Options:** Yes, No
62. **Question Category:** Confidentiality and Disclosure
    **Question:** Please detail any restrictions or conditions which apply to disclosure to Representatives. If the conditions found are only part of the accepted exceptions list (namely: declaring its confidential nature to whom it is being disclosed, information regarding price sensitivity, recipient's professional obligations to maintain confidentiality), you do not need to report them; just answer 'No'. Otherwise, detail the conditions.
    **Answer Options:** TextMultiline
63. **Question Category:** Confidentiality and Disclosure
    **Question:** Please set out the relevant definition of “Representative"
    **Answer Options:** TextMultiline
64. **Question Category:** Additional Borrowers
    **Question:** Does the agreement provide for the accession of Additional Borrowers?
    **Answer Options:** Yes, No
65. **Question Category:** Additional Borrowers
    **Question:** Does the accession of additional Borrowers require Lender Consent?
    **Answer Options:** Consent of Simple majority Lenders (over 50%), Consent of simple majority plus consent of affected lenders, Consent of Majority Lenders (66.666%), Consent of Super majority Lenders (75%), Consent of all Lenders, Silent, Consent of all affected Lenders, Consent of Majority (66.666%) plus consent of affected Lenders, Other, No Lender consent required
66. **Question Category:** Additional Borrowers
    **Question:** Please provide details of the relevant requirements
    **Answer Options:** Text
67. **Question Category:** Interest rate provisions
    **Question:** Does the facility contain an interest rate floor?
    **Answer Options:** Yes, No
68. **Question Category:** Interest rate provisions
    **Question:** Is the interest rate floor set at a rate other than zero?
    **Answer Options:** Yes, No
69. **Question Category:** Interest rate provisions
    **Question:** Please set out the relevant rate
    **Answer Options:** Text
70. **Question Category:** Interest rate provisions
    **Question:** Does the facility contain an interest rate cap?
    **Answer Options:** Yes, No
71. **Question Category:** Interest rate provisions
    **Question:** Please provide further details.
    **Answer Options:** Text
72. **Question Category:** Prepayment Fee
    **Question:** Is a charge or fee payable on prepayments (whether full or partial)?
    **Answer Options:** Yes, No
73. **Question Category:** Prepayment Fee
    **Question:** Prepayments permitted up to certain amount per year before charges?
    **Answer Options:** Yes, No
74. **Question Category:** Prepayment Fee
    **Question:** What is the threshold of prepayments permitted before charges are incurred?
    **Answer Options:** Text
75. **Question Category:** Prepayment Fee
    **Question:** Is there a date after which borrower can make prepayments without fee or penalty?
    **Answer Options:** Yes, No, Yes but unable to confirm date
76. **Question Category:** Prepayment Fee
    **Question:** Date after which borrower can make prepayments without fee or penalty
    **Answer Options:** Text
77. **Question Category:** Prepayment Fee
    **Question:** What prepayment fees are payable?
    **Answer Options:** Text
78. **Question Category:** Prepayment Fee
    **Question:** Please detail the relevant clause which governs the payment of a prepayment fee
    **Answer Options:** Text

**Final Instruction:**
Ensure the final output is a valid JSON array containing an object for **all** questions provided in the filtered list above, following all formatting and content instructions precisely. **Crucially, double-check that every object in the array contains all the required keys (`Question Number`, `Question Category`, `Question`, `Answer`, `Clause Reference`, `Clause Wording`, `Answer Justification`) before concluding.**
"""
# ... (Paste your SECTIONS_TO_RUN dict here) ...
SECTIONS_TO_RUN = {
    "agreement_details": (1, 4),
    # "eligibility": (5, 36),
    # "confidentiality": (37, 63),
    # "additional_borrowers": (64, 66),
    # "interest_rate_provisions": (67, 71),
    # "prepayment_interest": (72, 78)
}
# ... (Paste your system_instruction_text here) ...
system_instruction_text = """You are analysing a facility agreement to understand whether the asset can be included within a Significant Risk Transfer or not (or with conditions, requirements, or exceptions) at NatWest. Your output must be precise, factual, and directly supported by evidence from the provided document(s). You must answer with UK spelling, not US. (e.g. 'analyse' is correct while 'analyze' is not). Adhere strictly to the JSON schema provided, ensuring every object in the output array contains all required keys."""


# --- 3. Helper Function Definitions ---

# --- filter_prompt_by_section function ---
# (Keep as is - make sure it uses st.warning/st.error correctly)
def filter_prompt_by_section(initial_full_prompt, section):
    if section not in SECTIONS_TO_RUN:
        raise ValueError(f"Invalid section specified: {section}. Must be one of {list(SECTIONS_TO_RUN.keys())}")
    start_q, end_q = SECTIONS_TO_RUN[section]
    questions_start_marker = "**Questions to Answer:**"
    questions_end_marker = "**Final Instruction:**"
    try:
        start_index = initial_full_prompt.index(questions_start_marker)
        end_index = initial_full_prompt.index(questions_end_marker)
        prompt_header = initial_full_prompt[:start_index]
        full_questions_block = initial_full_prompt[start_index + len(questions_start_marker):end_index].strip()
        prompt_footer = initial_full_prompt[end_index:]
    except ValueError:
        raise ValueError("Could not find question block markers ('**Questions to Answer:**' or '**Final Instruction:**') in the main prompt text definition.")

    question_entries = re.split(r'\n(?=\s*\d+\.\s*?\*\*Question Category:)', full_questions_block)
    if not question_entries or len(question_entries) < 2:
         question_entries = re.split(r'\n(?=\d+\.\s)', full_questions_block)
         if not question_entries or len(question_entries) < 2:
              question_entries = re.split(r'\n(?=\*\*Question Category:)', full_questions_block)

    filtered_question_texts = []
    processed_q_nums = set()

    for entry in question_entries:
        entry = entry.strip()
        if not entry: continue
        match = re.match(r'^\s*(\d+)\.', entry)
        if match:
            q_num = int(match.group(1))
            if start_q <= q_num <= end_q:
                filtered_question_texts.append(entry)
                processed_q_nums.add(q_num)
        else:
            if "**Question Category:**" in entry or "**Question:**" in entry:
                 try: # Use try-except in case st commands fail early
                     st.warning(f"Could not parse question number from entry starting: {entry[:100]}...")
                 except Exception: pass # Ignore Streamlit errors if called too early during parsing

    expected_q_nums = set(range(start_q, end_q + 1))
    missing_q_nums = expected_q_nums - processed_q_nums
    if missing_q_nums:
         try:
             st.warning(f"Parsing might have missed expected question numbers in range {start_q}-{end_q}: {sorted(list(missing_q_nums))}")
         except Exception: pass

    if not filtered_question_texts:
        try:
            st.error(f"No questions found for section '{section}' in range {start_q}-{end_q}. Check prompt formatting.")
        except Exception: pass
        raise ValueError(f"Failed to extract questions for section '{section}'.")

    filtered_questions_string = "\n\n".join(filtered_question_texts)
    task_end_marker = "specified section."
    insert_pos = prompt_header.find(task_end_marker)
    if insert_pos != -1:
         insert_pos += len(task_end_marker)
         section_note = f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}). The list below contains ONLY these questions.\n"
         final_header = prompt_header[:insert_pos] + section_note + prompt_header[insert_pos:]
    else:
         final_header = prompt_header + f"\n\n**Current Focus:** Answering ONLY questions for the '{section.upper()}' section (Questions {start_q}-{end_q}).\n"
    final_prompt_for_api = f"{final_header}{questions_start_marker}\n\n{filtered_questions_string}\n\n{prompt_footer}"
    return final_prompt_for_api


# --- Add this function definition ---
@st.cache_data(show_spinner=False)
def render_pdf_page_to_image(_pdf_bytes, page_number, dpi=150):
    """Renders a specific PDF page (1-based) to a PNG image bytes."""
    if not _pdf_bytes or page_number < 1:
        return None
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        # PyMuPDF uses 0-based index
        page_index = page_number - 1
        if page_index < 0 or page_index >= doc.page_count:
            st.warning(f"Page number {page_number} is out of range (1-{doc.page_count}).")
            doc.close()
            return None

        page = doc.load_page(page_index)
        # Render page to a pixmap (image)
        pix = page.get_pixmap(dpi=dpi)
        doc.close()

        # Convert pixmap to PNG bytes
        img_bytes = pix.tobytes("png")
        return img_bytes
    except Exception as e:
        st.error(f"Error rendering PDF page {page_number}: {e}")
        return None

# --- validate_ai_data function ---
# (Keep as is - make sure it uses st.warning/st.error correctly)
def validate_ai_data(data, section_name):
    if not isinstance(data, list):
        st.error(f"Error: AI Data for validation in section '{section_name}' is not a list.")
        return None

    validated_data = []
    invalid_items_details = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            details = f"Item Index {index}: Not a dictionary."
            invalid_items_details.append(details)
            continue

        item_keys = set(item.keys())
        missing_keys = AI_REQUIRED_KEYS - item_keys

        if not missing_keys:
            valid_types = True
            if not isinstance(item.get("Question Number"), int):
                valid_types = False
                details = f"Q#{item.get('Question Number', 'Unknown')}: 'Question Number' is not an integer."
                invalid_items_details.append(details)

            if valid_types:
                validated_data.append(item)

        else:
            q_num_str = f"Q#{item.get('Question Number', 'Unknown')}"
            details = f"{q_num_str}: Missing keys: {sorted(list(missing_keys))}"
            invalid_items_details.append(details)

    if invalid_items_details:
        warning_message = f"AI Data Validation Summary [Section: {section_name}]: {len(validated_data)} valid items. Found {len(invalid_items_details)} invalid items/issues:\n"
        for detail in invalid_items_details[:5]:
            warning_message += f"- {detail}\n"
        if len(invalid_items_details) > 5:
            warning_message += f"- ...and {len(invalid_items_details) - 5} more issues."
        st.warning(warning_message)

    return validated_data

# --- generate_section_analysis function ---
# (Keep as is - make sure it uses st.info/st.success/st.warning/st.error correctly)
def generate_section_analysis(section, uploaded_file_ref):
    st.info(f"--- Starting analysis for section: {section} ---")
    if not uploaded_file_ref:
        st.error(f"Error: No valid uploaded file reference for section '{section}'.")
        return None

    model = genai.GenerativeModel(model_name=MODEL_NAME, system_instruction=system_instruction_text)
    generation_config = types.GenerationConfig(
        response_mime_type="application/json",
        response_schema=ai_response_schema_dict,
        temperature=0.0, top_p=0.05, top_k=1
    )
    final_validated_data = None

    for attempt in range(1, 1 + MAX_VALIDATION_RETRIES + 1):
        if attempt > 1:
             st.info(f"--- Validation Retry Attempt {attempt-1}/{MAX_VALIDATION_RETRIES} for '{section}' ---")
             time.sleep(RETRY_DELAY_SECONDS)
        try:
            prompt_for_api = filter_prompt_by_section(full_prompt_text, section)
            contents = [uploaded_file_ref, prompt_for_api]
            st.info(f"Attempt {attempt}: Generating content for section: {section}...")
            response = model.generate_content(contents=contents, generation_config=generation_config)
            parsed_ai_data = None; validated_ai_data = None

            if response.parts:
                full_response_text = response.text
                try:
                    parsed_ai_data = json.loads(full_response_text)
                    st.info(f"Attempt {attempt}: Parsed JSON. Validating...")
                    validated_ai_data = validate_ai_data(parsed_ai_data, section)
                    if validated_ai_data is not None and len(validated_ai_data) > 0:
                        st.success(f"Attempt {attempt}: Validation successful for section '{section}'.")
                        final_validated_data = validated_ai_data; break
                    elif validated_ai_data is None:
                         st.error(f"Attempt {attempt}: Critical validation error for section '{section}'.")
                         break
                    else:
                        st.warning(f"Attempt {attempt}: Validation failed (0 valid items) for section '{section}'.")
                except json.JSONDecodeError as json_err:
                    st.error(f"ERROR (Attempt {attempt}): Failed to parse JSON for section '{section}': {json_err}")
            else:
                block_reason = "Unknown"; block_message = "N/A"; safety_ratings_str = "N/A"
                try:
                    if response.prompt_feedback:
                        if response.prompt_feedback.block_reason: block_reason = response.prompt_feedback.block_reason.name
                        block_message = response.prompt_feedback.block_reason_message or "N/A"
                        if response.prompt_feedback.safety_ratings:
                             ratings = [f"{sr.category.name}: {sr.probability.name}" for sr in response.prompt_feedback.safety_ratings]
                             safety_ratings_str = ", ".join(ratings)
                except AttributeError: pass
                st.warning(f"Warning (Attempt {attempt}): Empty response for section '{section}'.")
                st.warning(f"--> Block Reason: {block_reason}, Message: {block_message}, Safety: {safety_ratings_str}")
        except types.generation_types.BlockedPromptException as bpe:
             st.error(f"ERROR (Attempt {attempt}): Prompt blocked for section '{section}'. Reason: {bpe}")
        except types.generation_types.StopCandidateException as sce:
             st.error(f"ERROR (Attempt {attempt}): Generation stopped for section '{section}'. Reason: {sce}")
        except Exception as e:
            st.error(f"ERROR (Attempt {attempt}): API/Processing error for section '{section}': {type(e).__name__} - {e}")
            st.error(f"Traceback: {traceback.format_exc()}")
        if final_validated_data is not None: break

    if final_validated_data is None:
         st.error(f"--- Validation FAILED for section '{section}' after {attempt} attempt(s). ---")
    st.info(f"--- Finished analysis attempts for section: {section} ---")
    return final_validated_data

# --- get_page_for_clause function ---
@st.cache_data(show_spinner=False)
def get_page_for_clause(_pdf_bytes, clause_ref, clause_wording):
    if not _pdf_bytes or clause_ref == 'N/A': return None
    page_num = None
    try:
        doc = fitz.open(stream=_pdf_bytes, filetype="pdf")
        search_term_ref = str(clause_ref).strip()
        if search_term_ref and search_term_ref != 'N/A':
             for page_index, page in enumerate(doc):
                  rect_list = page.search_for(search_term_ref, flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
                  if rect_list: page_num = page_index + 1; break
        if page_num is None and clause_wording and clause_wording != 'N/A':
            search_term_wording = str(clause_wording).strip()
            search_snippet = search_term_wording[:80]
            if len(search_snippet) > 10:
                for page_index, page in enumerate(doc):
                     rect_list = page.search_for(search_snippet, flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)
                     if rect_list: page_num = page_index + 1; break
        doc.close()
    except Exception as e:
        st.warning(f"Error searching PDF for '{clause_ref}': {e}")
        return None
    return page_num

# --- get_pdf_base64 function ---
@st.cache_data(show_spinner=False)
def get_pdf_base64(_pdf_bytes):
    try:
        base64_pdf = base64.b64encode(_pdf_bytes).decode('utf-8')
        return base64_pdf
    except Exception as e:
        st.error(f"Error encoding PDF to base64: {e}")
        return None


# --- 4. Initialize Session State (AFTER functions, BEFORE UI elements) ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'pdf_bytes' not in st.session_state:
    st.session_state.pdf_bytes = None
if 'pdf_base64' not in st.session_state:
    st.session_state.pdf_base64 = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'run_key' not in st.session_state: # Add a key to track analysis runs
    st.session_state.run_key = 0


# --- 5. Streamlit UI Logic ---

st.title("📄 Facility Agreement Analyzer with PDF Viewer")
st.markdown("Upload a PDF, analyze it, and click clause references to jump to the relevant page.")

# --- API Key Handling ---
api_key = st.secrets.get(API_KEY_ENV_VAR) or os.environ.get(API_KEY_ENV_VAR)
if not api_key:
    st.error(f"🛑 **Error:** Gemini API Key not found. Configure in Secrets.")
    st.stop()
else:
    try:
        genai.configure(api_key=api_key)
    except Exception as config_err:
        st.error(f"🛑 Error configuring Gemini API: {config_err}")
        st.stop()

# --- File Upload ---
# Use a consistent key for the uploader
uploaded_file_obj = st.file_uploader(
    "1. Upload Facility Agreement PDF",
    type="pdf",
    key=f"pdf_uploader_{st.session_state.run_key}" # Change key on new run to allow re-upload after analysis
)

# Check if a new file has been uploaded
new_file_uploaded = False
if uploaded_file_obj is not None:
    uploaded_bytes = uploaded_file_obj.getvalue()
    if uploaded_bytes != st.session_state.pdf_bytes:
        st.session_state.pdf_bytes = uploaded_bytes
        st.session_state.pdf_base64 = get_pdf_base64(st.session_state.pdf_bytes)
        st.session_state.analysis_results = None # Reset results
        st.session_state.processing_complete = False
        st.session_state.current_page = 1
        new_file_uploaded = True
        st.success(f"✅ File '{uploaded_file_obj.name}' loaded.")

# --- Analysis Trigger ---
if st.session_state.pdf_bytes is not None:
    # Disable button briefly after clicking or while processing
    analyze_disabled = st.session_state.processing_complete

    if st.button("✨ Analyze Document", key="analyze_button", disabled=analyze_disabled):
        # Reset state specifically for this run
        st.session_state.analysis_results = None
        st.session_state.processing_complete = False # Mark as processing
        st.session_state.current_page = 1
        st.session_state.run_key += 1 # Increment run key to allow re-upload later if needed

        run_start_time = datetime.now()
        run_timestamp_str = run_start_time.strftime("%Y-%m-%d %H:%M:%S")
        # Need the filename again if the object reference changed
        base_file_name = uploaded_file_obj.name if uploaded_file_obj else "uploaded_file"

        progress_bar = st.progress(0, text="Starting analysis...")
        status_text = st.empty()

        temp_dir = "temp_uploads"; os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, f"{run_start_time.strftime('%Y%m%d%H%M%S')}_{base_file_name}")
        gemini_uploaded_file_ref = None
        all_validated_data = []
        sections_with_errors = []
        overall_success = True

        try:
            status_text.info("💾 Saving file temporarily..."); progress_bar.progress(5, text="Saving...")
            with open(temp_file_path, "wb") as f: f.write(st.session_state.pdf_bytes)

            status_text.info("🚀 Uploading to Google Cloud..."); progress_bar.progress(10, text="Uploading...")
            gemini_uploaded_file_ref = genai.upload_file(path=temp_file_path)
            st.info(f"☁️ Gemini File: {gemini_uploaded_file_ref.name}")
            progress_bar.progress(15, text="Uploaded.")

            num_sections = len(SECTIONS_TO_RUN)
            progress_per_section = (80 - 15) / num_sections if num_sections > 0 else 0

            for i, section_name in enumerate(SECTIONS_TO_RUN.keys()):
                current_progress = int(15 + (i * progress_per_section))
                progress_bar.progress(current_progress, text=f"Analyzing {section_name} ({i+1}/{num_sections})...")
                status_text.info(f"⏳ Analyzing section: {section_name}...")

                section_validated_data = generate_section_analysis(section_name, gemini_uploaded_file_ref)

                if section_validated_data:
                    for item in section_validated_data:
                        item["File Name"] = base_file_name; item["Generation Time"] = run_timestamp_str
                    all_validated_data.extend(section_validated_data)
                elif section_validated_data is None:
                    overall_success = False; sections_with_errors.append(f"{section_name} (Crit Val Error)")
                else:
                    overall_success = False; sections_with_errors.append(f"{section_name} (Val Failed)")

            if all_validated_data:
                status_text.info("📄 Finding page numbers..."); progress_bar.progress(80, text="Finding pages...")
                num_results = len(all_validated_data)
                page_progress_increment = (95 - 80) / num_results if num_results > 0 else 0
                for idx, item in enumerate(all_validated_data):
                     item['Page'] = get_page_for_clause(st.session_state.pdf_bytes, item.get('Clause Reference'), item.get('Clause Wording'))
                     current_sub_progress = int(80 + (idx * page_progress_increment))
                     progress_bar.progress(min(current_sub_progress, 95), text=f"Finding page for Q#{item.get('Question Number', '')}...")

                st.session_state.analysis_results = pd.DataFrame(all_validated_data)
                missing_cols = [col for col in EXCEL_COLUMN_ORDER if col not in st.session_state.analysis_results.columns]
                for col in missing_cols: st.session_state.analysis_results[col] = None
                st.session_state.analysis_results = st.session_state.analysis_results[EXCEL_COLUMN_ORDER]

            progress_bar.progress(100, text="Analysis Complete!")
            status_text.info("🏁 Analysis and page finding complete.")
            st.session_state.processing_complete = True # Mark processing done for this run

        except Exception as main_err:
             st.error(f"❌ Unexpected error during analysis: {main_err}")
             st.error(traceback.format_exc())
             overall_success = False
             st.session_state.processing_complete = False # Allow retry on error
        finally:
            status_text.info("🧹 Cleaning up...")
            if gemini_uploaded_file_ref:
                try: genai.delete_file(name=gemini_uploaded_file_ref.name)
                except Exception: pass
            if os.path.exists(temp_file_path):
                try: os.remove(temp_file_path)
                except Exception: pass
            time.sleep(1) # Brief pause before clearing status
            status_text.empty()
            progress_bar.empty()

        # Trigger a rerun to update the display correctly after analysis
        st.rerun()

# --- Display Area (Results and PDF Viewer) ---
# This part only runs if results are available in session state

if st.session_state.analysis_results is not None:
    col1, col2 = st.columns([6, 4]) # Results | PDF

    with col1:
        st.subheader("📊 Analysis Results")
        df_results = st.session_state.analysis_results

        if df_results.empty:
            st.warning("No analysis results were generated or validated.")
        else:
            # Iterate and display results with clickable links/buttons
            for index, row in df_results.iterrows():
                # REMOVE 'key' from st.expander
                with st.expander(f"**Q{row['Question Number']}:** {row['Question']}"):
                    st.markdown(f"**Category:** {row.get('Question Category', 'N/A')}")
                    st.markdown(f"**Answer:** {row.get('Answer', 'N/A')}")

                    page_num = row.get('Page')
                    # Ensure page_num is an integer if not None
                    if page_num is not None:
                        try:
                            page_num = int(page_num)
                        except (ValueError, TypeError):
                            page_num = None # Treat non-integer page numbers as invalid

                    clause_ref = row.get('Clause Reference', 'N/A')

                    if page_num and clause_ref != 'N/A':
                        # Keep the key for the button, as it needs unique identification
                        button_key = f"page_button_{row['Question Number']}_{index}"
                        if st.button(f"**Clause Reference:** {clause_ref} (Go to Page {page_num})", key=button_key):
                            st.session_state.current_page = page_num
                            # Rerun is implicit after button press updates state
                    else:
                        st.markdown(f"**Clause Reference:** {clause_ref} (Page: N/A)")

                    st.markdown("**Clause Wording:**")
                    # Keep unique keys for text_areas if needed, though disabled ones might not strictly require them
                    st.text_area(f"wording_{index}", row.get('Clause Wording', ''), height=100, disabled=True, label_visibility="collapsed", key=f"wording_key_{index}")
                    st.markdown("**Answer Justification:**")
                    st.text_area(f"justification_{index}", row.get('Answer Justification', ''), height=75, disabled=True, label_visibility="collapsed", key=f"justification_key_{index}")

            # --- Download Button ---
            # (Download button code remains the same)
            st.markdown("---")
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_results.to_excel(writer, index=False, sheet_name='Analysis')
            excel_data = output.getvalue()
            display_file_name = uploaded_file_obj.name if uploaded_file_obj else "analysis"
            safe_base_name = re.sub(r'[^\w\s-]', '', display_file_name.split('.')[0]).strip()
            download_filename = f"Analysis_{safe_base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            st.download_button(
                label="📥 Download Full Results as Excel",
                data=excel_data, file_name=download_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="download_excel"
            )

# --- Replace the 'with col2:' block ---
    with col2:
        st.subheader("📄 Page Viewer")

        if st.session_state.pdf_bytes:
            # Get total pages for navigation limits (cache this if needed)
            try:
                doc = fitz.open(stream=st.session_state.pdf_bytes, filetype="pdf")
                total_pages = doc.page_count
                doc.close()
            except Exception:
                total_pages = 1 # Fallback

            # Get current page, ensure it's within bounds
            current_display_page = max(1, min(int(st.session_state.get('current_page', 1)), total_pages))

            # --- Navigation Buttons ---
            nav_cols = st.columns([1, 3, 1]) # Prev | Page Info | Next
            with nav_cols[0]:
                if st.button("⬅️ Prev", key="prev_page", disabled=(current_display_page <= 1)):
                    st.session_state.current_page = max(1, current_display_page - 1)
                    st.rerun() # Rerun to render the new page image
            with nav_cols[1]:
                st.markdown(f"<div style='text-align: center;'>Page {current_display_page} of {total_pages}</div>", unsafe_allow_html=True)
            with nav_cols[2]:
                if st.button("Next ➡️", key="next_page", disabled=(current_display_page >= total_pages)):
                    st.session_state.current_page = min(total_pages, current_display_page + 1)
                    st.rerun() # Rerun to render the new page image

            # --- Render and Display Image ---
            st.markdown("---") # Separator
            image_bytes = render_pdf_page_to_image(st.session_state.pdf_bytes, current_display_page)

            if image_bytes:
                 # Use columns to constrain width if needed, or use_column_width='auto'
                st.image(image_bytes, caption=f"Page {current_display_page}", use_column_width='always')
            else:
                st.warning(f"Could not render page {current_display_page}.")

        else:
            st.warning("Upload a PDF and run analysis to view pages.")

# Handle cases where PDF is loaded but analysis hasn't run or finished
elif st.session_state.pdf_bytes is not None and not st.session_state.processing_complete:
    st.info("PDF loaded. Click 'Analyze Document' above.")

# Initial state when no PDF is loaded
elif st.session_state.pdf_bytes is None:
    st.info("⬆️ Upload a PDF file to begin.")