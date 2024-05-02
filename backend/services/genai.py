from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class GeminiProcessor:
    
    def __init__(self,model_name,project):
        self.model = VertexAI(model_name = model_name, project = project)

    def generate_document_summary(self,documents: list, **args):
        chain_type = "map_reduce" if len(documents) > 10 else "stuff"

        chain = load_summarize_chain(
            llm = self.model,
            chain_type = chain_type,
            **args)
        return chain.run(documents)
    
    # Counting tokens of the documents
    def count_total_token(self, docs: list):
        temp_model = GenerativeModel("gemini-1.0-pro")
        total = 0
        logger.info("Counting total billable characters..")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_billable_characters
        return total        
    
    def get_model(self):
        return self.model
    

class YoutubeProcessor:
    def __init__(self,genai_processor : GeminiProcessor):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000, 
            chunk_overlap = 0
            )
        self.GeminiProcessor = genai_processor
        
    def retrieve_youtube_documents(self,video_url:str,verbose = False):
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info = True)
        docs = loader.load()
        result = self.text_splitter.split_documents(docs)
        
        author = result[0].metadata['author']
        length = result[0].metadata['length']
        title = result[0].metadata['title']
        total_size = len(result)
        total_billable_characters = self.GeminiProcessor.count_total_token(result)

        if verbose:            
            logger.info(f"Author : {author} , Length : {length} , Title : {title} , Total Size : {total_size} , Total Billable Characters : {total_billable_characters}")
        
        return result

    # Group Size changed to Sample Size
    def find_key_concepts(self,documents: list, sample_size: int = 0, verbose = False):
        if sample_size > len(documents) :
            raise ValueError("Group Size is larger than the number of documents")
        
        if sample_size == 0 :
            sample_size = len(documents) // 5
            if verbose:
                logging.info(f"No Sample Size specified. Setting the number of documents per sample as 5. Sample Size is {sample_size}")

        # Number of documents in each group :
        num_docs_per_group = len(documents) // sample_size + (len(documents) % sample_size > 0)

        # Check Thresholds for Response Quality : 
        if num_docs_per_group > 10 :
            raise ValueError("Each group has more than 10 documents and output quality will be degraded significantly. Increase the sample size parameter to reduce the number of documents per group.")
        elif num_docs_per_group > 5:
            logging.warn("Each group has more than 5 documents and output quality is likely to be degraded. Consider increasing the sample size")   

        # Splitting the Documents in chunks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range(0, len(documents),num_docs_per_group)]

        # Finding Key Concepts :
        batch_concepts = []
        batch_cost = 0

        logger.info("Finding Key Concepts ...")

        for group in tqdm(groups):
            group_content = ""
            for doc in group:
                group_content += doc.page_content

            # Prompt Template for finding Key Concepts :
            prompt = PromptTemplate(
                template = """
                Find and define key concepts or terms found in the text:
                {text}
                
                Respond in the following format as a JSON object without any backticks separating each concept with a comma:
                {{"concept": "definition", "concept": "definition", ...}}
                """,
                input_variables=["text"]
            )

            # Chain creation
            chain = prompt | self.GeminiProcessor.model

            # Chain Execution 
            output_concept = chain.invoke({"text":group_content})
            batch_concepts.append(output_concept)

            if verbose:
                
                total_input_char = len(group_content)
                total_input_cost = (total_input_char/1000) * 0.000125

                logging.info(f"Running chain on {len(group)} documents")
                logging.info(f"Total Input Characters : {total_input_char}")
                logging.info(f"Total Cost : $ {total_input_cost}")

                total_output_char = len(output_concept)
                total_output_cost = (total_output_char/1000) * 0.000375

                logging.info(f"Total Output Characters : {total_output_char}")
                logging.info(f"Total Cost : $ {total_output_cost}")

                batch_cost += total_input_cost + total_output_cost

                logging.info(f"Total Group Cost : $ {total_input_cost + total_output_cost}")

        
        # Convert each JSON String in batch_concept to a Python Dict
        concepts = [(concept) for concept in batch_concepts]
        print(f"{concepts}")
        # processed_concepts = [json.loads(concept) for concept in batch_concepts]
        processed_concepts = []
        for concept in batch_concepts:
            try:
                concept_json = json.loads(concept)
                processed_concepts.append(concept_json)
            except json.JSONDecodeError as e :
                print("Error decoding JSON", e)



        logging.info(f"Total Analysis Cost: {batch_cost}")
        return processed_concepts
