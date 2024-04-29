from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAI
from vertexai.generative_models import GenerativeModel
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
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
        logger.info("Counting total tokens..")
        for doc in tqdm(docs):
            total += temp_model.count_tokens(doc.page_content).total_tokens
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

        if verbose:            
            logger.info(f"Author : {author} , Length : {length} , Title : {title} , Total Size : {total_size}")
        
        return result
    
    def find_key_concepts(self,documents: list, group_size: int = 2):
        if group_size > len(documents) :
            raise ValueError("Group Size is larger than the number of documents")
        

        # Number of documents in each group :
        num_docs_per_group = len(documents) // group_size + (len(documents) % group_size > 0)

        # Splitting the Documents in chunks of size num_docs_per_group
        groups = [documents[i:i+num_docs_per_group] for i in range(0, len(documents),num_docs_per_group)]

        # Finding Key Concepts :
        batch_concepts = []

        logger.info("Finding Key Concepts ...")

        for group in tqdm(groups):
            group_content = ""
            for doc in group:
                group_content += doc.page_content

            # Prompt Template for finding Key Concepts :
            prompt = PromptTemplate(
                template = """ 
                Find and define key concepts or terms found in the text : 
                {text}
                Respond in the following format as a string separating each concept with a comma:
                "concept": "definition"
                """,
                input_variables = ["text"]
            )

            # Chain creation
            chain = prompt | self.GeminiProcessor.model

            # Chain Execution 
            concept = chain.invoke({"text":group_content})
            batch_concepts.append(concept)

        return batch_concepts
