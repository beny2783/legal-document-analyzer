import streamlit as st
import PyPDF2
import spacy
from spacy.tokens import Doc, Span  # Add this import
import re
import logging
from typing import Dict, List, Tuple, Set, Optional, Union
import networkx as nx
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentAnalyzer:
    def __init__(self):
        try:
            # Try to load the transformer model first
            try:
                self.nlp = spacy.load("en_core_web_trf")
                logger.info("Successfully loaded transformer model")
            except OSError:
                # Fallback to a smaller model if transformer model isn't available
                logger.warning("Transformer model not found, downloading smaller model...")
                os.system("python -m spacy download en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Successfully loaded smaller model")
            
            # Initialize relationship graph
            self.relationship_graph = nx.DiGraph()
            
            # Define clause relationships
            self.clause_relationships = {
                "depends_on": ["subject to", "conditional upon", "provided that"],
                "modifies": ["amends", "changes", "updates"],
                "excludes": ["excludes", "does not apply", "except"],
                "includes": ["includes", "comprises", "contains"],
                "references": ["refers to", "as defined in", "pursuant to"]
            }
            
            # Custom entity patterns for legal concepts
            self.legal_entities = {
                "OBLIGATION": ["shall", "must", "will be required to"],
                "PERMISSION": ["may", "is entitled to", "has the right to"],
                "PROHIBITION": ["shall not", "must not", "is prohibited from"],
                "CONDITION": ["if", "when", "subject to", "provided that"],
                "EXCEPTION": ["except", "unless", "excluding", "other than"],
                "DEFINITION": ["means", "refers to", "is defined as"]
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize NLP components: {e}")
            raise
        
        # Update use cases to include tax implications
        self.use_cases = {
            "founder_leaving": {
                "keywords": [
                    "leaver", "termination", "resignation", "shares", 
                    "transfer", "good leaver", "bad leaver",
                    "compulsory transfer", "share valuation",
                    "tax", "capital gains", "income tax", "hmrc",
                    "tax liability", "tax relief", "entrepreneurs relief",
                    "business asset disposal", "share buyback"
                ],
                "patterns": [
                    r"good\s+leaver",
                    r"bad\s+leaver",
                    r"compulsory\s+transfer",
                    r"transfer\s+of\s+shares",
                    r"leaver\s+provisions",
                    r"share\s+valuation",
                    r"termination\s+of\s+employment",
                    r"tax\s+implications",
                    r"capital\s+gains\s+tax",
                    r"income\s+tax",
                    r"tax\s+liability",
                    r"entrepreneurs?\s+relief",
                    r"business\s+asset\s+disposal",
                    r"share\s+buyback",
                    r"tax\s+clearance"
                ]
            },
            "founder_joining": {
                "keywords": [
                    "shares", "vesting", "new shareholder", 
                    "share issue", "subscription", "allotment",
                    "tax", "income tax", "employment tax",
                    "share scheme", "emi options", "tax advantaged",
                    "growth shares", "hurdle shares"
                ],
                "patterns": [
                    r"issue\s+of\s+shares",
                    r"vesting\s+schedule",
                    r"share\s+subscription",
                    r"new\s+shareholder",
                    r"share\s+allotment",
                    r"tax\s+implications",
                    r"employment\s+tax",
                    r"emi\s+options?",
                    r"growth\s+shares",
                    r"hurdle\s+shares",
                    r"tax\s+advantaged"
                ]
            },
            "tax_implications": {
                "keywords": [
                    "tax", "capital gains", "income tax", "corporation tax",
                    "hmrc", "relief", "allowance", "disposal", "acquisition",
                    "valuation", "market value", "consideration", "stamp duty",
                    "entrepreneurs relief", "business asset disposal",
                    "emi", "tax advantage", "share scheme"
                ],
                "patterns": [
                    r"tax\s+implications?",
                    r"capital\s+gains\s+tax",
                    r"income\s+tax",
                    r"corporation\s+tax",
                    r"tax\s+relief",
                    r"tax\s+allowance",
                    r"market\s+value",
                    r"stamp\s+duty",
                    r"hmrc\s+clearance",
                    r"tax\s+advantage(d)?",
                    r"share\s+scheme",
                    r"emi\s+options?"
                ]
            }
        }

        # Add practical scenarios that combine multiple aspects
        self.practical_scenarios = {
            "founder_leaving": {
                "key_questions": [
                    "What happens to your shares?",
                    "How will they be valued?",
                    "What are the tax implications?",
                    "What restrictions apply after leaving?",
                    "What approvals are needed?"
                ],
                "required_elements": [
                    "leaver provisions",
                    "share valuation",
                    "transfer process",
                    "tax considerations",
                    "post-exit restrictions"
                ],
                "action_items": [
                    "Board approval requirements",
                    "Valuation process",
                    "Tax clearances needed",
                    "Notice periods",
                    "Required documentation"
                ]
            },
            # ... similar for other scenarios
        }

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text content from uploaded PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            logger.info(f"Successfully extracted {len(text)} characters from PDF")
            logger.debug(f"First 500 characters of text: {text[:500]}")
            
            if not text.strip():
                logger.warning("Extracted text is empty or only whitespace")
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise

    def calculate_relevance_score(self, text: str, keyword: str, pattern: str = None) -> float:
        """Calculate relevance score based on multiple factors"""
        score = 0.0
        
        # Context relevance
        doc = self.nlp(text)
        context_relevance = sum(token.is_stop == False for token in doc) / len(doc)
        score += context_relevance * 0.3
        
        # Keyword prominence
        keyword_count = text.lower().count(keyword.lower())
        keyword_density = keyword_count / len(text.split())
        score += min(keyword_density * 100, 1.0) * 0.3
        
        # Position importance (earlier in document = more important)
        position_score = 1.0 - (text.find(keyword.lower()) / len(text))
        score += position_score * 0.2
        
        # Pattern matching strength
        if pattern:
            pattern_matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += min(pattern_matches * 0.1, 0.2)
        
        return min(score, 1.0)

    def analyze_document(self, text: str, use_case: str) -> List[Tuple[str, str, float]]:
        """Analyze document text and return relevant sections with explanations and relevance scores"""
        logger.info(f"Starting analysis for use case: {use_case}")
        logger.info(f"Document length: {len(text)} characters")
        
        doc = self.nlp(text)
        findings = []
        
        case_info = self.use_cases.get(use_case.lower(), {})
        if not case_info:
            logger.warning(f"No case info found for use case: {use_case}")
            return findings
            
        keywords = case_info.get("keywords", [])
        patterns = case_info.get("patterns", [])
        
        logger.info(f"Searching for {len(keywords)} keywords and {len(patterns)} patterns")

        # Improved text splitting logic
        # First try to split by double newlines
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # If we only got one paragraph, try single newlines
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        
        # If still only one paragraph, try splitting by periods
        if len(paragraphs) <= 1:
            paragraphs = [p.strip() + '.' for p in text.split('.') if p.strip()]

        logger.info(f"Document split into {len(paragraphs)} sections")
        logger.debug(f"First few sections: {paragraphs[:3]}")

        # Process chunks of text instead of single paragraphs
        chunk_size = 500  # characters
        current_chunk = ""
        
        for paragraph in paragraphs:
            current_chunk += paragraph + "\n"
            
            if len(current_chunk) >= chunk_size:
                # Process the current chunk
                self._analyze_chunk(current_chunk, keywords, patterns, findings, use_case)
                current_chunk = ""
        
        # Process any remaining text
        if current_chunk:
            self._analyze_chunk(current_chunk, keywords, patterns, findings, use_case)

        logger.info(f"Analysis complete. Found {len(findings)} relevant sections")

        # Sort findings by relevance score
        findings.sort(key=lambda x: x[2], reverse=True)
        
        # Limit to top N most relevant findings
        MAX_FINDINGS = 10
        return findings[:MAX_FINDINGS]

    def _analyze_chunk(self, text: str, keywords: List[str], patterns: List[str], 
                      findings: List[Tuple[str, str, float]], use_case: str):
        """Analyze a chunk of text for matches"""
        # Check for keyword matches
        for keyword in keywords:
            if keyword.lower() in text.lower():
                logger.info(f"Found keyword match: {keyword}")
                logger.debug(f"Matching text: {text[:200]}...")
                score = self.calculate_relevance_score(text, keyword)
                if score > 0.3:  # Minimum relevance threshold
                    explanation = self.generate_explanation(keyword, use_case)
                    findings.append((text.strip(), explanation, score))
                    break
        
        # Check for pattern matches
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Found pattern match: {pattern}")
                logger.debug(f"Matching text: {text[:200]}...")
                score = self.calculate_relevance_score(text, pattern, pattern)
                if score > 0.3:  # Minimum relevance threshold
                    explanation = self.generate_explanation(pattern, use_case)
                    findings.append((text.strip(), explanation, score))
                    break

    def generate_explanation(self, term: str, use_case: str, context: Dict = None) -> str:
        """Generate context-aware explanations using semantic understanding"""
        explanations = {
            # Founder leaving explanations
            "shares": "This section discusses share allocation and ownership, which is crucial when a founder leaves.",
            "good leaver": "Good leaver provisions define favorable terms for share valuation and transfer when leaving under approved circumstances.",
            "bad leaver": "Bad leaver provisions may result in less favorable terms for share valuation and transfer.",
            "vesting": "Vesting schedules determine how many shares you've earned based on your time with the company.",
            "transfer": "Share transfer provisions dictate how and when you can transfer your shares.",
            "valuation": "These provisions determine how your shares will be valued upon departure.",
            
            # Tax related explanations
            "tax": "This section has tax implications that may affect your position. Consider seeking professional tax advice.",
            "capital gains": "Capital Gains Tax may be payable on any profit from selling shares. The rate depends on various factors.",
            "income tax": "Income tax implications may arise from share-related benefits or disposals.",
            "entrepreneurs relief": "Business Asset Disposal Relief (formerly Entrepreneurs' Relief) could reduce Capital Gains Tax to 10% if you qualify.",
            "hmrc": "This involves HMRC considerations. Professional tax advice is recommended.",
            "clearance": "HMRC clearance may be required or advisable for certain transactions.",
            
            # Share scheme explanations
            "emi options": "Enterprise Management Incentive (EMI) options offer tax advantages for qualifying companies and employees.",
            "growth shares": "Growth shares can offer tax advantages but must be properly structured and valued.",
            "hurdle": "Hurdle shares only gain value once certain company performance targets are met.",
            
            # General provisions
            "board": "Board approval or decisions may be required for certain actions.",
            "notice": "Notice periods and formal procedures may need to be followed.",
            "confidentiality": "Confidentiality obligations may continue after departure.",
            "non-compete": "Non-compete restrictions may limit your future activities.",
            "documentation": "Formal documentation and agreements will need to be executed."
        }
        
        if context and "clause_structure" in context:
            # Use clause structure to provide more detailed explanations
            clause_info = context["clause_structure"]
            return self._generate_contextual_explanation(term, use_case, clause_info)
        
        # Clean up the term for matching
        clean_term = term.lower().strip()
        
        # Try to find the most relevant explanation
        for key, explanation in explanations.items():
            if key in clean_term:
                return explanation
        
        # Default explanation if no specific match is found
        return f"This section contains relevant information about {term} that may affect your position as a {use_case}. Review carefully for implications on your rights and obligations."

    def analyze_clause_structure(self, text: str) -> Dict:
        """Analyze the structure and relationships between clauses"""
        doc = self.nlp(text)
        
        # Extract clause hierarchy
        clause_tree = self._build_clause_tree(doc)
        
        # Identify relationships between clauses
        relationships = self._identify_relationships(doc)
        
        # Extract conditions and dependencies
        conditions = self._extract_conditions(doc)
        
        # Analyze semantic meaning
        semantics = self._analyze_semantics(doc)
        
        return {
            "tree": clause_tree,
            "relationships": relationships,
            "conditions": conditions,
            "semantics": semantics
        }

    def _build_clause_tree(self, doc: Doc) -> Dict:
        """Build a hierarchical tree of clauses"""
        tree = {"root": [], "children": {}}
        
        for sent in doc.sents:
            # Analyze sentence structure
            clause_info = self._analyze_sentence_structure(sent)
            
            # Find main and subordinate clauses
            main_clause = self._extract_main_clause(sent)
            sub_clauses = self._extract_subordinate_clauses(sent)
            
            # Build hierarchy
            if main_clause:
                tree["root"].append({
                    "text": main_clause.text,
                    "type": self._determine_clause_type(main_clause),
                    "structure": clause_info
                })
                
                if sub_clauses:
                    tree["children"][main_clause.text] = [
                        {
                            "text": sc.text,
                            "type": self._determine_clause_type(sc),
                            "relation": self._determine_relationship(main_clause, sc)
                        }
                        for sc in sub_clauses
                    ]
        
        return tree

    def _analyze_sentence_structure(self, sent: Span) -> Dict:
        """Analyze the grammatical structure of a sentence"""
        return {
            "subject": self._extract_subject(sent),
            "verb": self._extract_main_verb(sent),
            "object": self._extract_object(sent),
            "modifiers": self._extract_modifiers(sent)
        }

    def _identify_relationships(self, doc: Doc) -> List[Dict]:
        """Identify relationships between different clauses"""
        relationships = []
        
        for sent1 in doc.sents:
            for sent2 in doc.sents:
                if sent1 != sent2:
                    relation = self._find_relationship(sent1, sent2)
                    if relation:
                        relationships.append({
                            "source": sent1.text,
                            "target": sent2.text,
                            "type": relation,
                            "confidence": self._calculate_relationship_confidence(sent1, sent2)
                        })
        
        return relationships

    def _extract_conditions(self, doc: Doc) -> List[Dict]:
        """Extract conditional statements and their implications"""
        conditions = []
        
        for sent in doc.sents:
            if self._is_conditional(sent):
                conditions.append({
                    "condition": self._extract_condition_clause(sent),
                    "consequence": self._extract_consequence_clause(sent),
                    "type": self._determine_condition_type(sent)
                })
        
        return conditions

    def _analyze_semantics(self, doc: Doc) -> Dict:
        """Analyze semantic meaning of clauses"""
        return {
            "obligations": self._extract_obligations(doc),
            "permissions": self._extract_permissions(doc),
            "prohibitions": self._extract_prohibitions(doc),
            "definitions": self._extract_definitions(doc)
        }

    def _determine_clause_type(self, clause: Span) -> str:
        """Determine the type of a clause based on its content and structure"""
        # Implementation using pattern matching and linguistic analysis
        pass

    def _find_relationship(self, clause1: Span, clause2: Span) -> str:
        """Find the relationship between two clauses"""
        # Implementation using dependency parsing and semantic analysis
        pass

    def _calculate_relationship_confidence(self, clause1: Span, clause2: Span) -> float:
        """Calculate confidence score for relationship between clauses"""
        # Implementation using semantic similarity and linguistic features
        pass

    def _generate_contextual_explanation(self, term: str, use_case: str, clause_info: Dict) -> str:
        """Generate contextual explanations using clause structure"""
        # Implementation using semantic understanding and clause structure
        pass

    def analyze_practical_implications(self, text: str, scenario: str) -> Dict:
        """Analyze practical implications and required actions"""
        scenario_info = self.practical_scenarios.get(scenario, {})
        findings = {
            "key_findings": [],
            "missing_elements": [],
            "action_items": [],
            "risks": [],
            "next_steps": []
        }
        
        # Analyze each required element
        for element in scenario_info.get("required_elements", []):
            relevant_sections = self._find_relevant_sections(text, element)
            if relevant_sections:
                findings["key_findings"].append({
                    "element": element,
                    "content": relevant_sections,
                    "practical_impact": self._analyze_practical_impact(relevant_sections, element)
                })
            else:
                findings["missing_elements"].append(element)

        # Identify required actions
        findings["action_items"] = self._identify_required_actions(text, scenario)
        
        # Assess risks
        findings["risks"] = self._assess_risks(text, scenario)
        
        # Generate next steps
        findings["next_steps"] = self._generate_next_steps(findings)
        
        return findings

    def _analyze_practical_impact(self, sections: List[str], element: str) -> str:
        """Analyze the practical impact of provisions"""
        impacts = {
            "leaver provisions": "These provisions determine your rights and obligations when leaving the company.",
            "share valuation": "This defines how your shares will be valued upon departure.",
            "transfer process": "This outlines the steps you must follow to transfer your shares.",
            "tax considerations": "This affects your tax position and potential liabilities.",
            "post-exit restrictions": "These are the limitations on your activities after leaving."
        }
        return impacts.get(element, f"This affects how {element} will be handled.")

    def _identify_required_actions(self, text: str, scenario: str) -> List[str]:
        """Identify specific actions required"""
        common_actions = {
            "founder_leaving": [
                "Provide written notice of departure",
                "Request share valuation",
                "Seek tax advice on share disposal",
                "Review non-compete restrictions",
                "Prepare transfer documentation"
            ],
            "founder_joining": [
                "Review share vesting terms",
                "Complete share subscription agreement",
                "Consider tax implications of share acquisition",
                "Review shareholder agreement",
                "Complete necessary Companies House filings"
            ]
        }
        return common_actions.get(scenario, [])

    def _assess_risks(self, text: str, scenario: str) -> List[Dict]:
        """Assess potential risks and issues"""
        common_risks = {
            "founder_leaving": [
                {"description": "Bad leaver provisions may affect share value", "severity": "High"},
                {"description": "Non-compete restrictions may limit future opportunities", "severity": "Medium"},
                {"description": "Tax implications of share disposal", "severity": "High"},
                {"description": "Potential disputes over share valuation", "severity": "Medium"},
                {"description": "Confidentiality obligations", "severity": "Medium"}
            ],
            "founder_joining": [
                {"description": "Vesting schedule risks", "severity": "High"},
                {"description": "Share rights limitations", "severity": "Medium"},
                {"description": "Future dilution risks", "severity": "Medium"},
                {"description": "Tax implications of share acquisition", "severity": "High"},
                {"description": "Drag-along obligations", "severity": "Medium"}
            ]
        }
        return common_risks.get(scenario, [])

    def _generate_next_steps(self, findings: Dict) -> List[str]:
        """Generate practical next steps"""
        next_steps = [
            "1. Review all identified sections carefully",
            "2. Seek professional legal advice on key terms",
            "3. Consult with tax advisor on implications"
        ]
        
        # Add steps based on missing elements
        if findings["missing_elements"]:
            next_steps.append("4. Clarify the following undefined areas:")
            for element in findings["missing_elements"]:
                next_steps.append(f"   - {element}")
        
        # Add steps based on risks
        high_risks = [r for r in findings["risks"] if r["severity"] == "High"]
        if high_risks:
            next_steps.append("5. Address high-priority risks:")
            for risk in high_risks:
                next_steps.append(f"   - {risk['description']}")
        
        return next_steps

    def _find_relevant_sections(self, text: str, element: str) -> List[str]:
        """Find relevant sections for a given element"""
        # Convert element to search terms
        search_terms = {
            "leaver provisions": ["leaver", "good leaver", "bad leaver"],
            "share valuation": ["valuation", "fair value", "market value"],
            "transfer process": ["transfer", "transmission", "dispose"],
            "tax considerations": ["tax", "capital gains", "income tax"],
            "post-exit restrictions": ["non-compete", "restriction", "confidential"]
        }
        
        terms = search_terms.get(element, [element])
        
        # Find relevant paragraphs
        paragraphs = text.split('\n\n')
        relevant = []
        
        for para in paragraphs:
            if any(term.lower() in para.lower() for term in terms):
                relevant.append(para.strip())
        
        return relevant[:3]  # Return top 3 most relevant sections

    def _extract_main_clause(self, sent: Span) -> Optional[Span]:
        """Extract the main clause from a sentence"""
        # Basic implementation - can be enhanced
        return sent

    def _extract_subordinate_clauses(self, sent: Span) -> List[Span]:
        """Extract subordinate clauses from a sentence"""
        # Basic implementation - can be enhanced
        return []

    def _determine_relationship(self, main_clause: Span, sub_clause: Span) -> str:
        """Determine the relationship between clauses"""
        # Basic implementation - can be enhanced
        return "related"

    def _extract_subject(self, sent: Span) -> str:
        """Extract the subject of a sentence"""
        for token in sent:
            if "subj" in token.dep_:
                return token.text
        return ""

    def _extract_main_verb(self, sent: Span) -> str:
        """Extract the main verb of a sentence"""
        for token in sent:
            if token.pos_ == "VERB":
                return token.text
        return ""

    def _extract_object(self, sent: Span) -> str:
        """Extract the object of a sentence"""
        for token in sent:
            if "obj" in token.dep_:
                return token.text
        return ""

    def _extract_modifiers(self, sent: Span) -> List[str]:
        """Extract modifiers from a sentence"""
        return [token.text for token in sent if token.dep_ in ["amod", "advmod"]]

    def _is_conditional(self, sent: Span) -> bool:
        """Check if a sentence is conditional"""
        condition_markers = ["if", "when", "unless", "provided"]
        return any(marker in sent.text.lower() for marker in condition_markers)

    def _extract_condition_clause(self, sent: Span) -> str:
        """Extract the condition part of a conditional sentence"""
        # Basic implementation - can be enhanced
        return ""

    def _extract_consequence_clause(self, sent: Span) -> str:
        """Extract the consequence part of a conditional sentence"""
        # Basic implementation - can be enhanced
        return ""

    def _determine_condition_type(self, sent: Span) -> str:
        """Determine the type of condition"""
        # Basic implementation - can be enhanced
        return "general"

    def _extract_obligations(self, doc: Doc) -> List[str]:
        """Extract obligations from the document"""
        return []

    def _extract_permissions(self, doc: Doc) -> List[str]:
        """Extract permissions from the document"""
        return []

    def _extract_prohibitions(self, doc: Doc) -> List[str]:
        """Extract prohibitions from the document"""
        return []

    def _extract_definitions(self, doc: Doc) -> List[str]:
        """Extract definitions from the document"""
        return []

    # ... rest of your existing methods ... 