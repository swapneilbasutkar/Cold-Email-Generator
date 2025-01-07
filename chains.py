import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv("GROQ_API_KEY")

class Chain:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=None,
            timeout=None,
            max_retries=2
            )
        
    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing
            following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )

        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={'page_data':cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}

            ### INSTRUCTION:
            You are Swapneil, a business development executive at Lienpaws Technologies. Lienpaws Technologies is an AI and Software Consulting company dedicated to facilitating
            the seameless integration of business processes through automated tools.
            Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalibility,
            process optimization, cost reduction, and heightened overall efficiency.
            Your job is to write cold email to the client regarding the job mentioned above describing the capability of Lienpaws Technologies
            in fulfilling their needs.
            Also add the most relevent ones from the following links to showcase Lienpaws Technologies portfolio: {link_list}
            Remember you are Swapneil, BDE at Lienpaws Technologies.
            Do not provide a preamble.
            ### EMAIL (NO PREAMBLE):
            
            """
        )

        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description":str(job), "link_list": links})
        return res.content
    
if __name__ == "__main__":
    os.getenv("GROQ_API_KEY")
