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
        """Initialize the analyzer with better defaults and validation"""
        self.nlp = spacy.load("en_core_web_sm")
        
        # Define practical scenarios with more specific requirements
        self.practical_scenarios = {
            "founder_leaving": {
                "required_elements": [
                    "leaver provisions",
                    "vesting",
                    "share valuation",
                    "transfer process",
                    "post-exit restrictions",
                    "tax considerations"
                ],
                "critical_elements": ["leaver provisions", "vesting"],  # Must-have elements
                "validation_rules": {
                    "tenure": r"^\d+(\.\d+)?\s*(year|month)s?$",
                    "equity": r"^\d+(\.\d+)?%?$"
                }
            },
            "founder_joining": {
                "required_elements": [
                    "share rights",
                    "voting rights",
                    "vesting schedule",
                    "reserved matters",
                    "transfer restrictions"
                ],
                "critical_elements": ["share rights", "voting rights"],
                "validation_rules": {
                    "equity": r"^\d+(\.\d+)?%?$",
                    "joining_type": ["Co-founder with equal rights", "Co-founder with limited rights"]
                }
            }
        }

        # Initialize other components
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

    def analyze_document(self, text: str, use_case: str, context: Dict = None) -> List[Tuple[str, str, float]]:
        """Analyze document with comprehensive context awareness"""
        logger.info(f"Starting analysis for use case: {use_case} with context: {context}")
        
        # Adjust analysis based on context
        if context:
            self._adjust_analysis_parameters(use_case, context)
        
        doc = self.nlp(text)
        findings = []
        
        case_info = self.use_cases.get(use_case.lower(), {})
        if not case_info:
            logger.warning(f"No case info found for use case: {use_case}")
            return findings
        
        # Process document with context awareness
        paragraphs = self._split_text_into_paragraphs(text)
        
        # Process chunks with context
        chunk_size = 500
        current_chunk = ""
        
        for paragraph in paragraphs:
            current_chunk += paragraph + "\n"
            if len(current_chunk) >= chunk_size:
                self._analyze_chunk(current_chunk, case_info, findings, use_case, context)
                current_chunk = ""
        
        if current_chunk:
            self._analyze_chunk(current_chunk, case_info, findings, use_case, context)
        
        # Sort findings considering context
        findings.sort(key=lambda x: self._calculate_contextual_relevance(x[0], x[2], use_case, context), reverse=True)
        
        return findings[:10]  # Return top 10 most relevant findings

    def _adjust_analysis_parameters(self, use_case: str, context: Dict):
        """Adjust analysis parameters based on context"""
        if use_case == "founder_leaving":
            leaving_type = context.get("leaving_type", "")
            tenure = context.get("tenure", "")
            
            # Adjust keywords and patterns based on leaving type
            if leaving_type == "Termination":
                self.use_cases[use_case]["keywords"].extend([
                    "misconduct", "termination", "bad leaver",
                    "immediate effect", "cause", "summary dismissal"
                ])
                self.use_cases[use_case]["patterns"].extend([
                    r"terminat(ion|e)\s+for\s+cause",
                    r"summary\s+dismiss(al|ed)"
                ])
            
            # Add early departure considerations
            if tenure:
                try:
                    tenure_years = float(tenure.split()[0])
                    if tenure_years < 2:
                        self.use_cases[use_case]["keywords"].extend([
                            "vesting", "cliff", "acceleration",
                            "unvested", "forfeiture"
                        ])
                except ValueError:
                    pass
                
        elif use_case == "founder_joining":
            joining_type = context.get("joining_type", "")
            equity = context.get("equity", "")
            
            # Adjust for joining type
            if joining_type == "Co-founder with limited rights":
                self.use_cases[use_case]["keywords"].extend([
                    "voting rights", "class rights", "limited rights",
                    "non-voting", "observer rights"
                ])
            
            # Add minority protection considerations
            if equity:
                try:
                    equity_pct = float(equity.replace('%', ''))
                    if equity_pct > 25:
                        self.use_cases[use_case]["keywords"].extend([
                            "minority protection", "reserved matters",
                            "veto rights", "tag along", "drag along"
                        ])
                except ValueError:
                    pass

    def _calculate_contextual_relevance(self, text: str, base_score: float, use_case: str, context: Dict) -> float:
        """Calculate relevance score with context consideration"""
        if not context:
            return base_score
        
        context_multiplier = 1.0
        
        if use_case == "founder_leaving":
            # Increase relevance for termination-related sections if leaving type is termination
            if context.get("leaving_type") == "Termination" and any(word in text.lower() for word in ["terminate", "dismissal", "cause"]):
                context_multiplier *= 1.5
            
            # Increase relevance for vesting sections for early departures
            if context.get("tenure"):
                try:
                    tenure_years = float(context["tenure"].split()[0])
                    if tenure_years < 2 and any(word in text.lower() for word in ["vest", "cliff"]):
                        context_multiplier *= 1.3
                except ValueError:
                    pass
                
        elif use_case == "founder_joining":
            # Increase relevance for rights sections if joining with limited rights
            if context.get("joining_type") == "Co-founder with limited rights" and any(word in text.lower() for word in ["voting", "rights", "limited"]):
                context_multiplier *= 1.4
            
            # Increase relevance for protection clauses with significant equity
            if context.get("equity"):
                try:
                    equity_pct = float(context["equity"].replace('%', ''))
                    if equity_pct > 25 and any(word in text.lower() for word in ["protect", "minority", "reserved"]):
                        context_multiplier *= 1.3
                except ValueError:
                    pass
        
        return min(base_score * context_multiplier, 1.0)

    def _analyze_chunk(self, text: str, case_info: Dict, findings: List[Tuple[str, str, float]], use_case: str, context: Dict):
        """Analyze a chunk of text for matches"""
        # Check for keyword matches
        for keyword in case_info.get("keywords", []):
            if keyword.lower() in text.lower():
                logger.info(f"Found keyword match: {keyword}")
                logger.debug(f"Matching text: {text[:200]}...")
                score = self.calculate_relevance_score(text, keyword)
                if score > 0.3:  # Minimum relevance threshold
                    explanation = self.generate_explanation(keyword, use_case, context)
                    findings.append((text.strip(), explanation, score))
                    break
        
        # Check for pattern matches
        for pattern in case_info.get("patterns", []):
            if re.search(pattern, text, re.IGNORECASE):
                logger.info(f"Found pattern match: {pattern}")
                logger.debug(f"Matching text: {text[:200]}...")
                score = self.calculate_relevance_score(text, pattern, pattern)
                if score > 0.3:  # Minimum relevance threshold
                    explanation = self.generate_explanation(pattern, use_case, context)
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

    def analyze_practical_implications(self, text: str, scenario: str, context: Dict = None) -> Dict:
        """Analyze practical implications with specific context-based calculations"""
        logger.info(f"Analyzing practical implications for scenario: {scenario}")
        
        findings = {
            "key_findings": [],
            "missing_elements": [],
            "action_items": [],
            "risks": [],
            "next_steps": [],
            "context_specific_analysis": {}
        }
        
        # Get scenario requirements
        scenario_info = self.practical_scenarios.get(scenario, {})
        required_elements = scenario_info.get("required_elements", [])
        
        # Analyze key findings
        for element in required_elements:
            relevant_sections = self._find_relevant_sections(text, element)
            if relevant_sections:
                findings["key_findings"].append({
                    "element": element,
                    "content": relevant_sections[0],  # Most relevant section
                    "practical_impact": self._analyze_practical_impact(element, relevant_sections[0], context)
                })
            else:
                findings["missing_elements"].append(element)
        
        # Add action items based on context
        if context:
            if scenario == "founder_leaving":
                findings["action_items"] = [
                    f"Provide {context.get('leaving_type', 'departure')} notice",
                    "Document current shareholding",
                    "Calculate vested shares",
                    "Review transfer requirements",
                    "Seek tax advice on share disposal",
                    "Check post-exit restrictions"
                ]
                
                # Add specific risks based on leaving type
                if context.get("leaving_type") == "Termination":
                    findings["risks"].extend([
                        {"severity": "High", "description": "Risk of bad leaver classification"},
                        {"severity": "High", "description": "Potential loss of unvested shares"},
                        {"severity": "Medium", "description": "Valuation may be at nominal value"}
                    ])
                
                # Add tenure-specific risks
                if context.get("tenure"):
                    try:
                        tenure_months = float(context["tenure"].split()[0]) * 12
                        if tenure_months < 12:
                            findings["risks"].append({
                                "severity": "High",
                                "description": "Pre-cliff departure - risk of total share forfeiture"
                            })
                    except ValueError:
                        pass
        
        # Generate next steps
        findings["next_steps"] = self._generate_next_steps(findings, context)
        
        # Add context-specific analysis
        if context:
            if scenario == "founder_leaving":
                findings["context_specific_analysis"] = self._analyze_leaving_context(text, context)
            elif scenario == "founder_joining":
                findings["context_specific_analysis"] = self._analyze_joining_context(text, context)
        
        logger.info(f"Analysis complete. Found {len(findings['key_findings'])} key findings")
        return findings

    def _analyze_practical_impact(self, element: str, section: str, context: Dict = None) -> Dict:
        """Enhanced practical impact analysis with confidence scoring"""
        if context is None:
            context = {}
        
        impact = {
            "description": "",
            "severity": "Low",
            "confidence": 0.0,
            "supporting_evidence": []
        }
        
        # Process the section with NLP
        doc = self.nlp(section)
        
        # Extract key phrases and patterns
        key_phrases = self._extract_key_phrases(doc)
        patterns = self._identify_legal_patterns(doc)
        
        # Calculate confidence based on evidence
        evidence_count = len(key_phrases) + len(patterns)
        impact["confidence"] = min(0.3 + (evidence_count * 0.1), 1.0)
        
        # Add supporting evidence
        impact["supporting_evidence"].extend(key_phrases)
        
        # Determine severity and description based on context and evidence
        severity, description = self._evaluate_impact_severity(
            element, key_phrases, patterns, context
        )
        
        impact["severity"] = severity
        impact["description"] = description
        
        return impact

    def _generate_next_steps(self, findings: Dict, context: Dict) -> List[str]:
        """Generate practical next steps based on findings and context"""
        next_steps = []
        
        # Add steps for missing elements
        if findings["missing_elements"]:
            next_steps.append("1. Seek clarification on the following points:")
            for element in findings["missing_elements"]:
                next_steps.append(f"   - {element}")
        
        # Add steps for high-risk items
        high_risks = [r for r in findings["risks"] if r["severity"] == "High"]
        if high_risks:
            next_steps.append("2. Address high-priority risks:")
            for risk in high_risks:
                next_steps.append(f"   - {risk['description']}")
        
        # Add context-specific steps
        if context:
            if context.get("leaving_type") == "Termination":
                next_steps.extend([
                    "3. Document compliance with notice periods",
                    "4. Prepare for share valuation process",
                    "5. Review post-exit restrictions"
                ])
        
        # Add general steps if list is empty
        if not next_steps:
            next_steps = [
                "1. Review all identified sections carefully",
                "2. Seek professional legal advice",
                "3. Consult with tax advisor",
                "4. Document all decisions and communications"
            ]
        
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

    def _split_text_into_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs using multiple strategies"""
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
        
        return paragraphs

    def _extract_vesting_period(self, vesting_matches: List[str]) -> float:
        """Extract vesting period from document text"""
        # Default to 48 months (4 years) if not found
        default_period = 48.0
        
        for text in vesting_matches:
            # Look for patterns like "4 year vesting" or "48 month vesting"
            year_match = re.search(r'(\d+)\s*year\s*vest', text.lower())
            month_match = re.search(r'(\d+)\s*month\s*vest', text.lower())
            
            if year_match:
                return float(year_match.group(1)) * 12
            if month_match:
                return float(month_match.group(1))
        
        return default_period

    def _extract_cliff_period(self, vesting_matches: List[str]) -> float:
        """Extract cliff period from document text"""
        # Default to 12 months (1 year) if not found
        default_cliff = 12.0
        
        for text in vesting_matches:
            # Look for patterns like "1 year cliff" or "12 month cliff"
            year_match = re.search(r'(\d+)\s*year\s*cliff', text.lower())
            month_match = re.search(r'(\d+)\s*month\s*cliff', text.lower())
            
            if year_match:
                return float(year_match.group(1)) * 12
            if month_match:
                return float(month_match.group(1))
        
        return default_cliff

    def _determine_leaver_category(self, leaving_type: str, leaver_provisions: List[str]) -> str:
        """Determine leaver category based on leaving type and provisions"""
        if leaving_type == "Termination":
            # Look for bad leaver definitions
            if any("bad leaver" in text.lower() and "termination" in text.lower() 
                   for text in leaver_provisions):
                return "Bad Leaver"
        elif leaving_type == "Voluntary departure":
            # Look for intermediate leaver definitions
            if any("intermediate leaver" in text.lower() for text in leaver_provisions):
                return "Intermediate Leaver"
        
        # Default to good leaver if no specific provisions found
        return "Good Leaver"

    def _analyze_leaver_implications(self, leaving_type: str, leaver_provisions: List[str]) -> List[str]:
        """Analyze implications based on leaver type"""
        implications = []
        
        for provision in leaver_provisions:
            if leaving_type == "Termination" and "bad leaver" in provision.lower():
                implications.extend([
                    "Shares must be transferred within 30 days",
                    "No acceleration of unvested shares",
                    "Valuation likely at nominal value"
                ])
            elif "good leaver" in provision.lower():
                implications.extend([
                    "May retain vested shares",
                    "Possible acceleration of unvested shares",
                    "Valuation likely at fair market value"
                ])
        
        return list(set(implications))  # Remove duplicates

    def _extract_valuation_basis(self, leaving_type: str, leaver_provisions: List[str]) -> str:
        """Extract valuation basis from leaver provisions"""
        for provision in leaver_provisions:
            if leaving_type == "Termination" and "bad leaver" in provision.lower():
                if "nominal value" in provision.lower():
                    return "Nominal value per Bad Leaver provisions"
                elif "fair value" in provision.lower():
                    return "Fair market value despite Bad Leaver status"
            elif "good leaver" in provision.lower():
                if "fair value" in provision.lower() or "market value" in provision.lower():
                    return "Fair market value per Good Leaver provisions"
        
        # Default response if no specific valuation basis found
        return "Standard valuation process applies"

    def _analyze_leaving_context(self, text: str, context: Dict) -> Dict:
        """Analyze specific implications for leaving founder"""
        analysis = {
            "equity_calculation": {},
            "notice_requirements": {},
            "restrictions": {},
            "key_dates": {}
        }
        
        # Extract vesting information
        vesting_matches = self._find_relevant_sections(text, "vesting")
        vesting_period = self._extract_vesting_period(vesting_matches)
        cliff_period = self._extract_cliff_period(vesting_matches)
        
        # Calculate equity implications
        tenure = context.get("tenure", "")
        if tenure:
            try:
                tenure_months = float(tenure.split()[0]) * 12
                analysis["equity_calculation"] = {
                    "total_tenure_months": tenure_months,
                    "vesting_period_months": vesting_period,
                    "cliff_period_months": cliff_period,
                    "vested_percentage": self._calculate_vested_percentage(tenure_months, vesting_period, cliff_period)
                }
            except ValueError:
                analysis["equity_calculation"] = {
                    "error": "Could not calculate equity due to invalid tenure format"
                }
        
        # Extract notice requirements
        notice_sections = self._find_relevant_sections(text, "notice")
        if notice_sections:
            analysis["notice_requirements"] = {
                "notice_period": self._extract_notice_period(notice_sections),
                "special_requirements": self._extract_special_requirements(notice_sections)
            }
        
        # Extract post-exit restrictions
        restriction_sections = self._find_relevant_sections(text, "restriction")
        if restriction_sections:
            analysis["restrictions"] = {
                "non_compete": self._extract_non_compete_terms(restriction_sections),
                "non_solicit": self._extract_non_solicit_terms(restriction_sections),
                "confidentiality": self._extract_confidentiality_terms(restriction_sections)
            }
        
        return analysis

    def _calculate_vested_percentage(self, tenure_months: float, vesting_period: float, cliff_period: float) -> Dict:
        """Calculate vested percentage based on tenure and vesting schedule"""
        result = {
            "vested_percent": 0,
            "forfeited_percent": 100,
            "vesting_status": {
                "status": "",
                "impact": ""
            }
        }
        
        if tenure_months < cliff_period:
            result.update({
                "vesting_status": {
                    "status": "Pre-cliff period",
                    "impact": f"No shares vested - cliff period of {cliff_period} months not reached"
                }
            })
        elif tenure_months >= vesting_period:
            result.update({
                "vested_percent": 100,
                "forfeited_percent": 0,
                "vesting_status": {
                    "status": "Fully vested",
                    "impact": "All shares vested"
                }
            })
        else:
            vested_percent = (tenure_months / vesting_period) * 100
            result.update({
                "vested_percent": round(vested_percent, 2),
                "forfeited_percent": round(100 - vested_percent, 2),
                "vesting_status": {
                    "status": "Partially vested",
                    "impact": f"Vesting in progress - {round(vested_percent, 2)}% vested"
                }
            })
        
        return result

    def _extract_notice_period(self, notice_sections: List[str]) -> str:
        """Extract notice period requirements"""
        for section in notice_sections:
            # Look for patterns like "X months notice" or "X weeks notice"
            month_match = re.search(r'(\d+)\s*month[s]?\s*notice', section.lower())
            week_match = re.search(r'(\d+)\s*week[s]?\s*notice', section.lower())
            
            if month_match:
                return f"{month_match.group(1)} months"
            if week_match:
                return f"{week_match.group(1)} weeks"
        
        return "Standard notice period applies"

    def _extract_special_requirements(self, notice_sections: List[str]) -> List[str]:
        """Extract any special notice requirements"""
        requirements = []
        keywords = ["writing", "board", "approval", "immediate", "garden leave"]
        
        for section in notice_sections:
            for keyword in keywords:
                if keyword in section.lower():
                    requirements.append(f"Requires {keyword}")
        
        return requirements

    def _extract_non_compete_terms(self, sections: List[str]) -> Dict:
        """Extract non-compete restrictions"""
        return {
            "duration": self._extract_restriction_duration(sections, "non-compete"),
            "scope": self._extract_restriction_scope(sections, "non-compete")
        }

    def _extract_non_solicit_terms(self, sections: List[str]) -> Dict:
        """Extract non-solicitation restrictions"""
        return {
            "duration": self._extract_restriction_duration(sections, "non-solicit"),
            "scope": self._extract_restriction_scope(sections, "non-solicit")
        }

    def _extract_confidentiality_terms(self, sections: List[str]) -> Dict:
        """Extract confidentiality requirements"""
        return {
            "duration": "Indefinite",  # Usually confidentiality is indefinite
            "scope": self._extract_restriction_scope(sections, "confidential")
        }

    def _extract_restriction_duration(self, sections: List[str], restriction_type: str) -> str:
        """Extract duration of a specific restriction"""
        for section in sections:
            if restriction_type in section.lower():
                month_match = re.search(r'(\d+)\s*month[s]?', section.lower())
                year_match = re.search(r'(\d+)\s*year[s]?', section.lower())
                
                if month_match:
                    return f"{month_match.group(1)} months"
                if year_match:
                    return f"{year_match.group(1)} years"
        
        return "Duration not specified"

    def _extract_restriction_scope(self, sections: List[str], restriction_type: str) -> str:
        """Extract scope of a specific restriction"""
        for section in sections:
            if restriction_type in section.lower():
                # Return the relevant sentence containing the scope
                sentences = section.split('.')
                for sentence in sentences:
                    if restriction_type in sentence.lower():
                        return sentence.strip()
        
        return "Scope not explicitly defined"

    def _validate_context(self, scenario: str, context: Dict) -> Tuple[bool, List[str]]:
        """Validate context data against scenario requirements"""
        if scenario not in self.practical_scenarios:
            return False, ["Invalid scenario"]
        
        errors = []
        rules = self.practical_scenarios[scenario].get("validation_rules", {})
        
        for field, rule in rules.items():
            value = context.get(field)
            if not value:
                continue
            
            if isinstance(rule, str):  # Regex pattern
                if not re.match(rule, str(value)):
                    errors.append(f"Invalid {field} format")
            elif isinstance(rule, list):  # Enumerated values
                if value not in rule:
                    errors.append(f"Invalid {field} value")
        
        return len(errors) == 0, errors

    def _preprocess_context(self, context: Dict) -> Dict:
        """Standardize and clean context data"""
        if not context:
            return {}
        
        processed = {}
        
        # Standardize tenure format
        if "tenure" in context:
            tenure = context["tenure"]
            if "year" in tenure.lower():
                months = float(tenure.split()[0]) * 12
                processed["tenure_months"] = months
            elif "month" in tenure.lower():
                processed["tenure_months"] = float(tenure.split()[0])
        
        # Standardize equity format
        if "equity" in context:
            equity = context["equity"].replace("%", "")
            processed["equity_percentage"] = float(equity)
        
        # Copy other fields
        for key in ["leaving_type", "joining_type"]:
            if key in context:
                processed[key] = context[key]
        
        return processed

    def _check_data_quality(self, findings: Dict) -> Dict:
        """Check quality and completeness of analysis"""
        quality_metrics = {
            "completeness": 0.0,
            "confidence": 0.0,
            "missing_critical": [],
            "warnings": []
        }
        
        # Check for critical elements
        scenario = findings.get("scenario", "")
        if scenario in self.practical_scenarios:
            critical_elements = self.practical_scenarios[scenario]["critical_elements"]
            found_elements = {f["element"] for f in findings.get("key_findings", [])}
            
            missing_critical = set(critical_elements) - found_elements
            if missing_critical:
                quality_metrics["missing_critical"] = list(missing_critical)
                quality_metrics["warnings"].append(
                    "Critical information missing - analysis may be incomplete"
                )
        
        # Calculate completeness
        total_elements = len(self.practical_scenarios.get(scenario, {}).get("required_elements", []))
        found_elements = len(findings.get("key_findings", []))
        quality_metrics["completeness"] = found_elements / total_elements if total_elements > 0 else 0
        
        # Calculate average confidence
        confidences = [
            f.get("practical_impact", {}).get("confidence", 0)
            for f in findings.get("key_findings", [])
        ]
        quality_metrics["confidence"] = sum(confidences) / len(confidences) if confidences else 0
        
        findings["quality_metrics"] = quality_metrics
        return findings

    def _extract_key_phrases(self, doc: Doc) -> List[str]:
        """Extract key legal phrases from text"""
        key_phrases = []
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if any(word in chunk.text.lower() for word in [
                "shares", "rights", "obligations", "restrictions", 
                "notice", "termination", "vesting"
            ]):
                key_phrases.append(chunk.text.strip())
        
        # Extract verb phrases with legal significance
        for token in doc:
            if token.pos_ == "VERB" and any(word in token.text.lower() for word in [
                "shall", "must", "will", "agree", "terminate", "vest"
            ]):
                phrase = self._get_verb_phrase(token)
                if phrase:
                    key_phrases.append(phrase.strip())
        
        return list(set(key_phrases))  # Remove duplicates

    def _get_verb_phrase(self, verb_token) -> str:
        """Extract complete verb phrase from a verb token"""
        phrase = []
        
        # Get subject
        for child in verb_token.children:
            if "subj" in child.dep_:
                phrase.append(child.text)
        
        # Add verb
        phrase.append(verb_token.text)
        
        # Get object and other important children
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj", "attr"]:
                phrase.append(child.text)
        
        return " ".join(phrase)

    def _identify_legal_patterns(self, doc: Doc) -> List[str]:
        """Identify common legal patterns in text"""
        patterns = []
        
        # Look for obligation patterns
        for token in doc:
            if token.text.lower() in ["shall", "must", "will"]:
                patterns.append("obligation")
            elif token.text.lower() in ["may", "can", "entitled"]:
                patterns.append("permission")
            elif token.text.lower() in ["unless", "except", "provided"]:
                patterns.append("condition")
        
        # Look for definition patterns
        for sent in doc.sents:
            if any(word in sent.text.lower() for word in ["means", "defined", "refers to"]):
                patterns.append("definition")
        
        return list(set(patterns))

    def _evaluate_impact_severity(self, element: str, key_phrases: List[str], patterns: List[str], context: Dict) -> Tuple[str, str]:
        """Evaluate severity and description of impact"""
        severity = "Low"
        description = f"This {element} may affect your rights"
        
        # Check for critical patterns
        if "obligation" in patterns:
            severity = "High"
            description = f"This {element} creates mandatory obligations"
        elif "condition" in patterns:
            severity = "Medium"
            description = f"This {element} contains important conditions"
        
        # Context-specific evaluation
        if context:
            if element == "leaver provisions" and context.get("leaving_type") == "Termination":
                severity = "High"
                description = "Critical for determining share treatment on termination"
            elif element == "vesting" and context.get("tenure"):
                try:
                    tenure_months = float(context["tenure"].split()[0]) * 12
                    if tenure_months < 12:
                        severity = "High"
                        description = "Critical for determining vested equity"
                except ValueError:
                    pass
        
        return severity, description

    # ... rest of your existing methods ... 