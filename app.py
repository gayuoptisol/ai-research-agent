import asyncio
from gpt_researcher import GPTResearcher
import re
import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator
from typing import List, Optional
from datetime import date, datetime
from prettytable import PrettyTable

# Load environment variables
load_dotenv()

from langchain_openai import ChatOpenAI
import os

llm = ChatOpenAI(
    openai_api_base="https://api.openai.com/v1",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model_name="gpt-4o-mini",
)


class ContactInformation(BaseModel):
    email: Optional[str] = Field(None, description="Company email address")
    phone: Optional[str] = Field(None, description="Company phone number")
    website: Optional[str] = Field(None, description="Company website URL")

    @validator("email")
    def validate_email(cls, v):
        if v is None or v.strip() == "":
            return None
        if v and "@" in v and "." in v:
            return v
        return None

    @validator("website")
    def validate_website(cls, v):
        if v is None or v.strip() == "":
            return None
        if v and ("http://" in v or "https://" in v):
            return v
        return None


class CompanyInformation(BaseModel):
    primary_address: str = Field(
        ..., description="Complete registered address of the company"
    )
    registration_number: Optional[str] = Field(
        None, description="Company registration/identification number"
    )
    legal_form: str = Field(..., description="Legal structure of the company")
    country: str = Field(..., description="Country where company is registered")
    town: str = Field(..., description="City and state/province of registration")
    registration_date: str = Field(..., description="Date of company incorporation")
    contact_information: ContactInformation = Field(
        ..., description="Company contact details"
    )
    general_details: str = Field(..., description="Brief description of the company")
    ubo: Optional[str] = Field(None, description="Ultimate Business Owners information")
    directors_shareholders: List[str] = Field(
        ..., description="List of directors and shareholders"
    )
    subsidiaries: Optional[str] = Field(
        None, description="Information about company subsidiaries"
    )
    parent_company: Optional[str] = Field(
        None, description="Parent company information if any"
    )
    last_reported_revenue: str = Field(
        ..., description="Latest reported revenue information"
    )

    @validator("registration_number")
    def validate_registration_number(cls, v):
        if v is None or v.strip() == "":
            return "Information not available"
        # Remove common separators and clean up
        cleaned = re.sub(r"[^a-zA-Z0-9]", "", v)
        return cleaned if cleaned else "Information not available"

    @validator("directors_shareholders", pre=True)
    def parse_directors(cls, v):
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            directors = [d.strip() for d in re.split(r"[,;]|\band\b", v) if d.strip()]
            return directors or ["Information not available"]
        return ["Information not available"]


def sanitize_string(s: str) -> str:
    if not s:
        return "Information not available"
    return s.strip() or "Information not available"


async def generate_evidence(query: str):
    try:
        researcher = GPTResearcher(
            query=query, report_type="research_report", config_path=None
        )
        await researcher.conduct_research()
        report = await researcher.write_report()

        split_text = report.split("## References")
        main_text = split_text[0].strip()

        if "## Conclusion" in main_text:
            main_text = main_text.split("## Conclusion")[0].strip()

        citation_pattern = r"\[(.*?)\]\((https?://\S+)\)"
        citations = re.findall(citation_pattern, main_text)

        references = "\n".join(
            [
                f"- {source.strip()} {url.strip().replace(')', '')}"
                for source, url in citations
            ]
        )

        main_text = re.sub(citation_pattern, "", main_text).strip()

        if len(split_text) > 1:
            references += "\n" + split_text[1].strip()

        return main_text, references
    except Exception as e:
        st.error(f"Error during research: {str(e)}")
        return "", "No references available due to error"


def format_company_data_as_dict(company_info):
    try:
        directors_str = ", ".join(company_info.directors_shareholders)

        all_fields = {
            "Fields": [
                "Primary Address",
                "Registration Number",
                "Legal Form",
                "Country",
                "Town",
                "Registration Date",
                "Email",
                "Phone",
                "Website",
                "General Details",
                "Directors & Shareholders",
                "UBO",
                "Subsidiaries",
                "Parent Company",
                "Last Reported Revenue",
            ],
            "Details": [
                sanitize_string(company_info.primary_address),
                sanitize_string(company_info.registration_number),
                sanitize_string(company_info.legal_form),
                sanitize_string(company_info.country),
                sanitize_string(company_info.town),
                company_info.registration_date,
                sanitize_string(company_info.contact_information.email),
                sanitize_string(company_info.contact_information.phone),
                sanitize_string(str(company_info.contact_information.website)),
                sanitize_string(company_info.general_details),
                directors_str,
                sanitize_string(company_info.ubo),
                sanitize_string(company_info.subsidiaries),
                sanitize_string(company_info.parent_company),
                sanitize_string(company_info.last_reported_revenue),
            ],
        }
        return all_fields
    except Exception as e:
        st.error(f"Error formatting company data: {str(e)}")
        return {
            "Fields": ["Error"],
            "Details": ["Failed to format company information"],
        }


def final_output_generation(llm, report):
    try:
        structured_llm = llm.with_structured_output(CompanyInformation)
        result = structured_llm.invoke(report)
        return result
    except Exception as e:
        st.error(f"Error processing company information: {str(e)}")
        return CompanyInformation(
            primary_address="Information not available",
            registration_number="Information not available",
            legal_form="Information not available",
            country="Information not available",
            town="Information not available",
            registration_date=date(1900, 1, 1),
            contact_information=ContactInformation(),
            general_details="Information not available",
            directors_shareholders=["Information not available"],
            last_reported_revenue="Information not available",
        )


def main():
    st.title("AI Web Research Agent")

    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Enter the name of the company:")

    with col2:
        country = st.text_input("Enter country (optional):")

    if st.button("Fetch Details"):
        if not company_name:
            st.warning("Please enter a company name.")
            return

        try:
            with st.spinner("Fetching details, please wait..."):
                if country.strip():
                    query = f"Provide the {company_name} details specifically in {country}, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Focus on the company's operations, registration number, and registration in {country}."
                else:
                    query = f"Provide the {company_name} details, including registration number, primary address, legal form, country, town, registration date, contact info, general details, UBO, directors/shareholders, subsidiaries, parent company, and last reported revenue. Make sure to include any company registration or identification numbers."

                report, references = asyncio.run(generate_evidence(query=query))
                result = final_output_generation(llm, report)

                data_dict = format_company_data_as_dict(result)

                st.subheader("AI Web Research Agent Result: ")
                st.table(data_dict)

                st.subheader("References")
                st.markdown(references, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.warning(
                "Please try again with a different company name or check your internet connection."
            )


if __name__ == "__main__":
    main()
