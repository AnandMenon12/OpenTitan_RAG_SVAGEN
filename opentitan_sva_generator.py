import os
import re
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime

import requests
from bs4 import BeautifulSoup
import markdown
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

@dataclass
class DocumentChunk:
    """Represents a chunk of documentation with metadata"""
    content: str
    ip_name: str
    file_path: str
    section: str
    signal_refs: List[str]
    fsm_state_refs: List[str]
    chunk_type: str  # 'doc', 'register', 'rtl'
    metadata: Dict[str, Any]

@dataclass
class RTLSignal:
    """Represents an RTL signal or port"""
    name: str
    width: int
    direction: str  # input, output, internal
    module: str
    reset_value: Optional[str] = None
    description: Optional[str] = None

@dataclass
class RegisterField:
    """Represents a register field from HJSON"""
    name: str
    address: str
    reset_value: str
    access: str
    brief: str
    register_name: str

class OpenTitanIngester:
    """Handles ingestion of OpenTitan documentation and RTL"""
    
    def __init__(self, opentitan_root: str):
        self.opentitan_root = Path(opentitan_root)
        self.target_ips = ['uart', 'i2c', 'kmac', 'lc_ctrl', 'otbn', 'sysrst_ctrl']
        
    def get_ip_paths(self, ip_name: str) -> Dict[str, Path]:
        """Get relevant paths for an IP"""
        ip_base = self.opentitan_root / "hw" / "ip" / ip_name
        return {
            'rtl': ip_base / "rtl",
            'doc': ip_base / "doc",
            'data': ip_base / "data"
        }
    
    def _fetch_web_content(self, url: str, ip_name: str) -> List[DocumentChunk]:
        """Fetch and parse content from a web URL."""
        print(f"Fetching documentation from {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find the main content area of the OpenTitan book
            main_content = soup.find('main')
            if not main_content:
                print("Could not find main content in the page.")
                return []

            return self._parse_html_content(main_content, ip_name, url)
        except requests.RequestException as e:
            print(f"Error fetching URL {url}: {e}")
            return []

    def _parse_html_content(self, content_soup: BeautifulSoup, ip_name: str, file_path: str) -> List[DocumentChunk]:
        """Parse HTML content into structured chunks."""
        chunks = []
        current_section = "Overview"
        
        for element in content_soup.find_all(['h1', 'h2', 'h3', 'h4', 'p', 'pre', 'ul', 'ol', 'table']):
            if element.name in ['h1', 'h2', 'h3', 'h4']:
                current_section = element.get_text().strip()
            else:
                text = element.get_text().strip()
                if len(text) > 20: # Simple filter for meaningful content
                    chunks.append(DocumentChunk(
                        content=text,
                        ip_name=ip_name,
                        file_path=file_path,
                        section=current_section,
                        signal_refs=self._extract_signal_refs(text),
                        fsm_state_refs=self._extract_fsm_refs(text),
                        chunk_type='doc',
                        metadata={'source': 'web'}
                    ))
        return chunks

    def extract_markdown_content(self, md_file: Path) -> List[DocumentChunk]:
        """Convert markdown to structured chunks"""
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return self._parse_html_content(soup, md_file.parent.parent.name, str(md_file))
    
    def _extract_signal_refs(self, text: str) -> List[str]:
        """Extract signal references from text"""
        return list(set(re.findall(r'\b[a-z_]+(?:_i|_o|_en|_req|_ack|_gnt|_rdy|_valid|_p)\b', text)))
    
    def _extract_fsm_refs(self, text: str) -> List[str]:
        """Extract FSM state references from text"""
        return list(set(re.findall(r'\b[A-Z_]+_STATE\b|\bState[A-Z][a-z_]+\b', text)))
    
    def parse_hjson_registers(self, hjson_file: Path) -> List[RegisterField]:
        """Parse HJSON register definitions"""
        try:
            import hjson
        except ImportError:
            print("hjson not available, skipping register parsing")
            return []
        
        fields = []
        
        try:
            with open(hjson_file, 'r') as f:
                data = hjson.load(f)
            
            if 'registers' in data:
                for reg in data.get('registers', []):
                    reg_name = reg.get('name', 'unknown')
                    for field in reg.get('fields', []):
                        fields.append(RegisterField(
                            name=field.get('name', ''),
                            address=f"0x{reg.get('offset', '0x0')}",
                            reset_value=str(field.get('resval', 0)),
                            access=field.get('access', 'ro'),
                            brief=field.get('desc', ''),
                            register_name=reg_name
                        ))
        except Exception as e:
            print(f"Error parsing {hjson_file}: {e}")
        
        return fields
    
    def extract_rtl_symbols(self, rtl_dir: Path) -> List[RTLSignal]:
        """Extract RTL symbols using simple parsing"""
        signals = []
        
        for sv_file in rtl_dir.glob("*.sv"):
            try:
                with open(sv_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                module_match = re.search(r'module\s+([a-zA-Z0-9_]+)', content)
                module_name = module_match.group(1) if module_match else sv_file.stem
                
                # A more robust regex for ports
                port_regex = re.compile(
                    r'\b(input|output|inout)\s+'  # Direction
                    r'(?:logic|wire|reg)?\s*'      # Type
                    r'(?:\[[^\]]+\])?\s*'         # Packed dimensions
                    r'([a-zA-Z0-9_]+)\s*'         # Port name
                    r'(?=[,);])',                # Lookahead for comma, paren, or semicolon
                    re.MULTILINE
                )

                for match in port_regex.finditer(content):
                    signals.append(RTLSignal(
                        name=match.group(2),
                        width=1,  # Simplified
                        direction=match.group(1),
                        module=module_name
                    ))
                
            except Exception as e:
                print(f"Error parsing {sv_file}: {e}")
        
        return signals
    
    def ingest_ip(self, ip_name: str) -> Tuple[List[DocumentChunk], List[RegisterField], List[RTLSignal]]:
        """Ingest all data for a specific IP"""
        paths = self.get_ip_paths(ip_name)
        
        # Ingest documentation
        doc_chunks = []
        if ip_name == 'i2c':
            doc_chunks = self._fetch_web_content(
                "https://opentitan.org/book/hw/ip/i2c/doc/theory_of_operation.html", 
                'i2c'
            )
        elif paths['doc'].exists():
            for md_file in paths['doc'].glob("*.md"):
                doc_chunks.extend(self.extract_markdown_content(md_file))
        
        # Parse registers
        register_fields = []
        if paths['data'].exists():
            for hjson_file in paths['data'].glob(f"{ip_name}*.hjson"):
                register_fields.extend(self.parse_hjson_registers(hjson_file))
        
        # Extract RTL symbols
        rtl_signals = []
        if paths['rtl'].exists():
            rtl_signals = self.extract_rtl_symbols(paths['rtl'])
        
        return doc_chunks, register_fields, rtl_signals

class EmbeddingManager:
    """Manages embeddings and FAISS indexes"""
    
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.indexes = {}  # ip_name -> faiss index
        self.chunks = {}   # ip_name -> list of chunks
        
    def create_embeddings(self, chunks: List[DocumentChunk]) -> np.ndarray:
        """Create embeddings for document chunks"""
        if not chunks:
            return np.array([])
        texts = [f"{chunk.section}: {chunk.content}" for chunk in chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings
    
    def build_faiss_index(self, ip_name: str, chunks: List[DocumentChunk]):
        """Build FAISS index for an IP"""
        if not chunks:
            print(f"No chunks to index for {ip_name}")
            return
        
        embeddings = self.create_embeddings(chunks)
        if embeddings.size == 0:
            print(f"Embedding creation failed for {ip_name}")
            return

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)
        
        self.indexes[ip_name] = index
        self.chunks[ip_name] = chunks
    
    def search(self, ip_name: str, query: str, k: int = 10) -> List[Tuple[DocumentChunk, float]]:
        """Search for relevant chunks"""
        if ip_name not in self.indexes:
            return []
        
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        
        scores, indices = self.indexes[ip_name].search(query_embedding, k)
        
        return [(self.chunks[ip_name][idx], float(score)) for score, idx in zip(scores[0], indices[0]) if idx != -1]

class SVAGenerator:
    """Generates SVA properties using Qwen/Qwen2-7B-Instruct"""
    
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """Load the Qwen2-7B-Instruct model"""
        print(f"Loading {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        print("Model loaded successfully")
    
    def build_context(self, chunks: List[Tuple[DocumentChunk, float]], 
                     registers: List[RegisterField], 
                     signals: List[RTLSignal]) -> str:
        """Build context from retrieved information"""
        context_parts = ["=== CONTEXT FOR SVA GENERATION ==="]
        
        if chunks:
            context_parts.append("\n--- Relevant Documentation Sections ---")
            for chunk, score in chunks[:6]:
                context_parts.append(f"Section: {chunk.section} (Relevance: {score:.2f})")
                context_parts.append(f"Content: {chunk.content}\n")
        
        if registers:
            context_parts.append("\n--- Relevant Register Fields ---")
            for reg in registers[:10]:
                context_parts.append(
                    f"- {reg.register_name}.{reg.name}: "
                    f"addr={reg.address}, rst={reg.reset_value}, acc={reg.access}, desc='{reg.brief}'"
                )
        
        if signals:
            context_parts.append("\n--- Relevant RTL Signals ---")
            for sig in signals[:15]:
                context_parts.append(f"- {sig.direction} {sig.name} (in module {sig.module})")
        
        return "\n".join(context_parts)
    
    def generate_sva_properties(self, ip_name: str, context: str, query: str) -> str:
        """Generate SVA properties using Qwen2-72B-Instruct"""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        system_prompt = """You are a world-class expert in SystemVerilog Assertions (SVA) for semiconductor IP verification. Your task is to generate precise, high-quality SVA properties based on the provided context.

Follow these rules strictly:
1.  **Property Block:** Enclose every assertion in a named `property` block.
2.  **Assertion:** Follow each property with a corresponding `assert property` statement.
3.  **Clocking:** Use `@(posedge clk_i)` for clocking.
4.  **Reset:** Use `disable iff (!rst_ni)` for asynchronous reset.
5.  **Comments:** Add a brief, insightful comment above each property explaining its purpose.
6.  **No Placeholders:** Do not use placeholder signals. Only use signals found in the context.
7.  **Focus:** Generate properties directly related to the user's query and the provided context.

Example of a perfect SVA property:
```systemverilog
// Ensures that a request is eventually followed by a grant.
property req_followed_by_gnt;
  @(posedge clk_i) disable iff (!rst_ni)
  req |=> ##[1:$] gnt;
endproperty
assert_req_gnt: assert property (req_followed_by_gnt);
```
"""

        user_prompt = f"""Based on the following context for the `{ip_name}` IP, generate SVA properties for the request: "{query}"

{context}

Generate SVA properties now.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=2048,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return content

class OpenTitanSVASystem:
    """Main system orchestrating the entire pipeline"""
    
    def __init__(self, opentitan_root: str, cache_dir: str = "./cache"):
        self.opentitan_root = opentitan_root
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        self.ingester = OpenTitanIngester(opentitan_root)
        self.embedding_manager = EmbeddingManager()
        self.sva_generator = SVAGenerator()
        
        self.ip_data = {}
    
    def _get_cache_path(self, ip_name: str) -> Path:
        return self.cache_dir / f"{ip_name}_ingested_data.pkl"

    def ingest_and_cache_ip(self, ip_name: str):
        """Ingest data for a single IP and cache it."""
        print(f"Ingesting data for {ip_name}...")
        doc_chunks, register_fields, rtl_signals = self.ingester.ingest_ip(ip_name)
        
        self.ip_data[ip_name] = {
            'doc_chunks': doc_chunks,
            'register_fields': register_fields,
            'rtl_signals': rtl_signals
        }
        
        with open(self._get_cache_path(ip_name), 'wb') as f:
            pickle.dump(self.ip_data[ip_name], f)
            
        print(f"Ingested and cached {ip_name}: {len(doc_chunks)} doc chunks, "
              f"{len(register_fields)} registers, {len(rtl_signals)} signals")

    def load_ip_from_cache(self, ip_name: str) -> bool:
        """Load IP data from cache."""
        cache_file = self._get_cache_path(ip_name)
        if not cache_file.exists():
            return False
        
        print(f"Loading cached data for {ip_name}...")
        with open(cache_file, 'rb') as f:
            self.ip_data[ip_name] = pickle.load(f)
        return True

    def prepare_ip(self, ip_name: str):
        """Prepare an IP for use, loading from cache or ingesting."""
        if ip_name not in self.ip_data:
            if not self.load_ip_from_cache(ip_name):
                self.ingest_and_cache_ip(ip_name)
        
        # Always build the FAISS index after loading/ingesting
        self.embedding_manager.build_faiss_index(ip_name, self.ip_data[ip_name]['doc_chunks'])

    def generate_sva_for_query(self, ip_name: str, query: str) -> str:
        """Generate SVA properties for a specific query about an IP"""
        if ip_name not in self.ingester.target_ips:
            return f"Error: IP '{ip_name}' is not a valid target."

        self.prepare_ip(ip_name)
        
        # Retrieve relevant context
        doc_results = self.embedding_manager.search(ip_name, query, k=10)
        
        # Filter signals and registers based on query terms
        query_terms = set(re.findall(r'\w+', query.lower()))
        
        relevant_signals = [
            s for s in self.ip_data[ip_name]['rtl_signals'] 
            if any(term in s.name.lower() for term in query_terms)
        ] or self.ip_data[ip_name]['rtl_signals']

        relevant_registers = [
            r for r in self.ip_data[ip_name]['register_fields']
            if any(term in r.name.lower() or term in r.register_name.lower() for term in query_terms)
        ] or self.ip_data[ip_name]['register_fields']
        
        # Build context
        context = self.sva_generator.build_context(doc_results, relevant_registers, relevant_signals)
        
        # Generate SVA properties
        sva_code = self.sva_generator.generate_sva_properties(ip_name, context, query)
        
        # Log the interaction
        self.log_interaction(ip_name, query, context, sva_code)
        
        return sva_code
    
    def log_interaction(self, ip_name: str, query: str, context: str, output: str):
        """Logs the generation context and output to a JSON file."""
        log_dir = self.cache_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{ip_name}_{timestamp}.json"
        
        log_data = {
            "timestamp": timestamp,
            "ip_name": ip_name,
            "query": query,
            "model_name": self.sva_generator.model_name,
            "context": context,
            "generated_sva": output
        }
        
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        print(f"Interaction logged to {log_file}")
    
    def interactive_session(self):
        """Run single session for generating SVA properties"""
        print("\n--- OpenTitan SVA Generator ---")
        print(f"Available IPs: {', '.join(self.ingester.target_ips)}")
        
        try:
            ip_name = input("Enter IP name: ").strip().lower()
            if not ip_name:
                print("No IP name provided. Exiting.")
                return
            
            if ip_name not in self.ingester.target_ips:
                print(f"Unknown IP. Please choose from: {', '.join(self.ingester.target_ips)}")
                return
            
            query = input(f"Query for {ip_name}: ").strip()
            if not query:
                print("No query provided. Exiting.")
                return
            
            print("\nGenerating SVA properties... (this may take a moment)")
            sva_code = self.generate_sva_for_query(ip_name, query)
            print("\n" + "="*80)
            print("GENERATED SVA PROPERTIES:")
            print(sva_code)
            print("="*80 + "\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main entry point"""
    opentitan_root = "/home/eng/a/axm200085/Transformer_annotated/NeuripsEDU/opentitan"
    
    system = OpenTitanSVASystem(opentitan_root)
    system.interactive_session()

if __name__ == "__main__":
    # This ensures that the dataclasses are defined in the opentitan_sva_generator module
    # which prevents pickle errors when loading cached data in other scripts.
    main()

