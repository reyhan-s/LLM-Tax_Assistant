
import os
import re
import uuid
import shutil
import warnings
import json
import time 
from typing import List, Dict, Any, Optional, Iterator
from difflib import SequenceMatcher
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain_core.documents import Document
from langchain_core.stores import BaseStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

warnings.filterwarnings("ignore")

def log_persian_output(content: str, filename: str = "search_results.txt"):
    """Writes complex Persian output to a file to prevent terminal corruption."""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(content + "\n\n")

class SimpleDocStore(BaseStore):
    def __init__(self):
        self._store = {}
        
    def mget(self, keys: List[str]) -> List[Optional[Document]]:
        return [self._store.get(key) for key in keys]
        
    def mset(self, key_value_pairs: List[tuple[str, Document]]) -> None:
        for key, value in key_value_pairs:
            self._store[key] = value

    def mdelete(self, keys: List[str]) -> None:
        for key in keys:
            if key in self._store:
                del self._store[key]

    def yield_keys(self, prefix: Optional[str] = None) -> Iterator[str]:
        keys = list(self._store.keys())
        if prefix:
            keys = [key for key in keys if key.startswith(prefix)]
        
        for key in keys:
            yield key

#  TEXT NORMALIZER 
class TextNormalizer:
    @staticmethod
    def process(text: str) -> str:
        text = text.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        text = text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"[#*>]", " ", text)
        text = text.replace("ي", "ی").replace("ك", "ک")
        text = re.sub(r"\s+", " ", text)
        return text.strip()


# METADATA EXTRACTOR
class MetadataExtractor:
    @staticmethod
    def _normalize(text: str) -> str:
        t = text.replace("ي", "ی").replace("ك", "ک")
        t = t.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))
        t = t.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def extract(filename: str, text: str) -> Dict[str, Any]:
        
        try:
            clean = MetadataExtractor._normalize(text)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

            meta = {
                "source": filename,
                "doc_id": filename, # temp ID
                "category": "مالیات مستقیم",
                "subject": "",
                "year": "",
                "number": "",
                "related_laws": ""
            }

            if (m := re.search(r"### شماره:\s*(.+)", text)):
                doc_number = m.group(1).strip()
                meta["number"] = doc_number
                meta["doc_id"] = doc_number # FIX: stable key

            if (m := re.search(r"### موضوع:\s*\*?(.*)", text)):
                meta["subject"] = m.group(1).strip()
            else:
                for line in lines[:3]:
                    if 5 <= len(line) <= 150:
                        meta["subject"] = line
                        break

            if (m := re.search(r"### تاریخ:\s*(\d{4})/\d{1,2}/\d{1,2}", text)):
                meta["year"] = m.group(1)

            t = clean.lower()
            if "ارزش افزوده" in t or "vat" in t or "عوارض" in t:
                meta["category"] = "مالیات بر ارزش افزوده"
            elif any(k in t for k in ["حقوق", "دستمزد", "کارکنان", "مالیات حقوق"]):
                meta["category"] = "مالیات حقوق"
            elif any(k in t for k in ["ماده 169", "جرایم", "ابلاغ", "اعتراض"]):
                meta["category"] = "مقررات اجرایی"

            laws = []
            for law_match in re.findall(r"\[([^\]]+)\]\((https?://inta\.tax\.gov\.ir/Pages/Action/LawsShow/[^\)]+)\)", text):
                law_name, law_link = law_match
                laws.append({law_name: law_link})
            if laws:
                meta["related_laws"] = json.dumps(laws, ensure_ascii=False)

            return meta
        
        except Exception as e:
            # logs extraction error to terminal
            print(f"⚠️ Extraction Error in file {filename}: {e}")
            return {
                "source": filename, "doc_id": str(uuid.uuid4())[:8], 
                "category": "نامشخص", "subject": "خطای استخراج", "year": "", "number": "", "related_laws": ""
            }


#  SPLITTER FACTORY
class SplitterFactory:
    @staticmethod
    def get_child_splitter(embeddings):
        return RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


#  INTENT ROUTER
class IntentRouter:
    categories = {
        "مالیات بر ارزش افزوده": ["ارزش افزوده", "VAT", "عوارض"],
        "مالیات حقوق": ["حقوق کارکنان", "مطالبه حقوق", "کسر مالیات", "اعتراض به مالیات حقوق"],
        "مقررات اجرایی": ["جرایم", "ابلاغ", "اعتراض", "کمیسیون", "هیأت حل اختلاف"],
        "مالیات مستقیم": ["اظهارنامه", "عملکرد", "سود", "ترازنامه"],
    }

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.vectors = {c: np.mean(self.embeddings.embed_documents(kws), axis=0)
                        for c, kws in self.categories.items()}

    def route(self, query: str):
        qv = self.embeddings.embed_query(query)
        best_cat, best_score = None, -1
        q_norm = qv / np.linalg.norm(qv) if np.linalg.norm(qv) else qv
        
        for cat, vec in self.vectors.items():
            vec_norm = vec / np.linalg.norm(vec) if np.linalg.norm(vec) else vec
            score = np.dot(q_norm, vec_norm)
            if score > best_score:
                best_cat, best_score = cat, score
        return best_cat, float(best_score)


#  TAX ASSISTANT ENGINE 
class TaxAssistantEngine:
    def __init__(self, folder):
        try:
            print("Initializing Embeddings...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}, 
                encode_kwargs={'normalize_embeddings': True} 
            )
        except Exception as e:
             print(f"ERROR: Embedding Initialization Failed: {e}")
             raise e
             
        self.router = IntentRouter(self.embeddings)
        self.docs_cache = []
        self.retriever = None
        self.vectorstore = None
        self.store = None
        self.data_folder = folder
        self._ingest()

    def _ingest(self):
        if not os.path.exists(self.data_folder):
            print(f"ERROR: Data folder not found: {self.data_folder}")
            return
        db_path = "./chroma_db"
        
        if os.path.exists(db_path):
            try:
                shutil.rmtree(db_path)
            except PermissionError:
                print("WARNING: Close other processes using ./chroma_db and try running as administrator.")
                return

        for name in os.listdir(self.data_folder):
            if name.endswith(".md"):
                path = os.path.join(self.data_folder, name)
                try:
                    text = open(path, "r", encoding="utf-8").read()
                    clean = TextNormalizer.process(text)
                    meta = MetadataExtractor.extract(name, text)
                    self.docs_cache.append(Document(page_content=clean, metadata=meta))
                except Exception as e:
                    print(f"WARNING: Failed to process file {name}: {e}")
                    
        if not self.docs_cache:
            print("WARNING: No documents found.")
            return

        child = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        parent = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)

        self.vectorstore = Chroma(collection_name="tax_docs", embedding_function=self.embeddings, persist_directory=db_path)
        self.store = SimpleDocStore() 

        self.retriever = ParentDocumentRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            child_splitter=child,
            parent_splitter=parent,
        )
        self.retriever.add_documents(self.docs_cache)
        print(f"INFO: Successfully indexed {len(self.docs_cache)} documents.")

    def search(self, query: str, filt: Dict[str, str] = None):
        """Main search method (calls benchmark_search)."""
        results, status, avg_time, child_count = self.benchmark_search(query, filt)
        return results, status

    def _search(self, query: str, filt: Dict[str, str] = None, k: int = 10):
        """Internal search logic with filtering and fallback."""
        filter_expr = None
        
        # 1. Intent Routing
        cat, score = self.router.route(query)
        if not filt and score > 0.9:
            filter_expr = {"category": cat}

        # 2. Metadata Filtering
        if filt and self.retriever:
            pre_filtered_docs = self.docs_cache
            
            # A. Exact Filtering
            for key, val in filt.items():
                if key != "subject":
                    pre_filtered_docs = [
                        d for d in pre_filtered_docs 
                        if str(d.metadata.get(key, "")).strip() == val.strip()
                    ]

            # B. Substring Filtering (Subject)
            if subject_val := filt.get("subject"):
                subject_val_lower = subject_val.strip().lower()
                pre_filtered_docs = [
                    d for d in pre_filtered_docs 
                    if subject_val_lower in d.metadata.get("subject", "").strip().lower()
                ]

            if not pre_filtered_docs:
                return [], "No Meta Match"
            
            #  Exact Match Fallback
            if len(pre_filtered_docs) == 1:
                return [pre_filtered_docs[0]], "Direct Meta Hit"
            
            # C. Final Chroma Filter Construction
            ids = [d.metadata["doc_id"] for d in pre_filtered_docs]
            filter_expr = {"_source_id": {"$in": ids}} 


        # 3. Final Retrieval
        if filter_expr and self.retriever:
            child_hits = self.vectorstore.similarity_search(query, k=k, filter=filter_expr)
            parent_ids = list(set(d.metadata.get("_source_id") for d in child_hits))
            
            if parent_ids:
                return [d for d in self.store.mget(parent_ids) if d], "Filtered"
            else:
                return self._search(query, filt=None, k=k) # Fallback to Full Search
        
        return self.retriever.invoke(query), "Full Search"

    def benchmark_search(self, query: str, filt: Dict[str, str] = None, runs: int = 5):
        """Runs search multiple times to calculate average latency and search space."""
        import time 
        times = []
        results = []
        
        for _ in range(runs):
            start_time = time.time()
            res, status = self._search(query, filt) 
            end_time = time.time()
            
            times.append(end_time - start_time)
            results = res
            
        avg_time = sum(times) / runs
        
        # Calculate approximate children count for benchmarking report
        if filt:
            pre_filtered_docs = [d for d in self.docs_cache]
            for key, val in filt.items():
                if key != "subject":
                    pre_filtered_docs = [d for d in pre_filtered_docs if str(d.metadata.get(key, "")).strip() == val.strip()]
            
            approx_children_count = len(pre_filtered_docs) * 5 # Assuming 5 children per parent for estimation
        else:
            approx_children_count = len(self.docs_cache) * 5 
            
        return results, status, avg_time, approx_children_count

    @staticmethod
    def _fuzzy_match(a: str, b: str, threshold: float = 0.5):
        """Fuzzy match logic (retained but not used in final search logic)."""
        return SequenceMatcher(None, a.strip().lower(), b.strip().lower()).ratio() >= threshold


def display_menu():
    print("\n" + "="*40)
    print("Tax Assistant Engine")
    print("="*40)
    print("1. 🔎 Search (Query/Filter)")
    print("2. 📂 View Documents (Metadata)")
    print("3. ❌ Exit")
    print("="*40)

def format_persian_output(d, status, avg_time, child_count):
    """Formats the output string in Persian for file logging."""
    if d:
        output = f"\n--- Analytical Search Results ---"
        output += f"\nQuery Status: {status}"
        output += f"\nAvg. Time ({5} runs): {avg_time:.4f} sec"
        output += f"\nSearch Space: {child_count} Child Chunks"
        output += "\n" + "-"*40
        output += f"\n✅ منبع: {d.metadata.get('source')}"
        output += f"\nسال: {d.metadata.get('year')}"
        output += f"\nدسته: {d.metadata.get('category')}"
        output += f"\nموضوع: {d.metadata.get('subject')}"
        output += f"\nشماره: {d.metadata.get('number')}"
        output += "\n" + "-"*40
        output += "\n--- Document Content ---"
        output += "\n" + d.page_content[:600] + "..."
        return output
    else:
        return f"\n--- Analytical Search Results ---\n⚠️ نتیجه‌ای یافت نشد. وضعیت: {status}"

def main():
    data_path = "./data"
    
    try:
        engine = TaxAssistantEngine(folder=data_path)
    except Exception as e:
        print(f"FATAL ERROR: Initialization failed: {e}")
        return

    if not engine.docs_cache or not engine.retriever:
        print("WARNING: System initialization failed or no documents found.")
        return
    
    valid_fields = ["year", "category", "subject", "number"]
    
    while True:
        display_menu()
        choice = input("Enter option: ").strip()
        
        if choice == "1":
            query = input("❓ Enter Query: ").strip()
            filt = {}

            if input("Add filter? (y/n): ").lower() == "y":
                print("Select filter fields:")
                for i, field in enumerate(valid_fields, 1):
                    print(f"{i}. {field}")
                
                sel_fields_raw = input("Enter field numbers (e.g., 1,3): ").split(",")
                
                try:
                    sel_fields = [int(idx.strip()) for idx in sel_fields_raw if idx.strip().isdigit()]
                    
                    for idx in sel_fields:
                        if 1 <= idx <= len(valid_fields):
                            field = valid_fields[idx - 1]
                            val = input(f"Value for {field}: ").strip()
                            if val:
                                filt[field] = val
                        else:
                             print(f"WARNING: Invalid number {idx} ignored.")

                except ValueError:
                    print("WARNING: Invalid field numbers input.")
                    continue

            results, status, avg_time, child_count = engine.benchmark_search(query, filt)
            
            persian_output = format_persian_output(results[0] if results else None, status, avg_time, child_count)
            log_persian_output(persian_output, filename="Tax_Assistant_Report.txt")
            
            print(f"\n--- Analytical Search Results ---")
            print(f"Status: {status}")
            print(f"Avg. Time ({5} runs): {avg_time:.4f} sec")
            print(f"Search Space Reduction: {child_count} Child Chunks")
            print("--- Output saved to Tax_Assistant_Report.txt ---")
            
            input("\nPress Enter to continue...")

        elif choice == "2":
            #  add metadata in file 
            doc_output = "\n--- Full Document Metadata List ---\n"
            for d in engine.docs_cache:
                doc_output += f"Source: {d.metadata.get('source')}, Year: {d.metadata.get('year')}, Subject: {d.metadata.get('subject')}\n"
            log_persian_output(doc_output, filename="Tax_Assistant_Report.txt")
            print("INFO: Document list metadata saved to Tax_Assistant_Report.txt")
            input("\nPress Enter to continue...")

        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("WARNING: Invalid option. Try again.")


if __name__ == "__main__":
    main()
