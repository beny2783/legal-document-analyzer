import streamlit as st
from document_analyzer import DocumentAnalyzer
import logging
from typing import Dict

# Set up logging
logger = logging.getLogger(__name__)

@st.cache_resource
def init_analyzer():
    try:
        return DocumentAnalyzer()
    except Exception as e:
        st.error(f"Failed to initialize analyzer: {str(e)}")
        return None

def main():
    st.title("Legal Document Analyzer")
    st.write("Analyze legal documents for specific scenarios")

    # Initialize analyzer with error handling
    analyzer = init_analyzer()
    if analyzer is None:
        st.error("Could not initialize the document analyzer. Please try again later.")
        return

    # File upload section
    st.subheader("1. Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file (e.g., Articles of Association)", type="pdf")

    # Use case selection with multi-select
    st.subheader("2. Select Scenarios")
    scenarios = st.multiselect(
        "What scenarios would you like to analyze?",
        options=[
            "Founder Leaving",
            "New Founder Joining",
        ],
        help="You can select multiple scenarios for comprehensive analysis"
    )
    
    # Additional context based on selected scenarios
    if scenarios:
        with st.expander("Additional Context (Optional)"):
            if "Founder Leaving" in scenarios:
                st.markdown("### Leaving Founder Details")
                st.text_input("How long has the founder been with the company?", key="tenure")
                st.radio("Leaving circumstances:", 
                        ["Voluntary departure", "Mutual agreement", "Termination"],
                        key="leaving_type")
                
            if "New Founder Joining" in scenarios:
                st.markdown("### New Founder Details")
                st.text_input("Expected equity percentage:", key="equity")
                st.radio("Joining as:", 
                        ["Co-founder with equal rights", "Co-founder with limited rights", "Technical co-founder"],
                        key="joining_type")

    # Add analyze button
    analyze_button = st.button("Analyze Document", 
                             disabled=not (uploaded_file and scenarios),
                             help="Upload a document and select scenarios to enable analysis")

    # Process document if button is clicked
    if analyze_button:
        try:
            with st.spinner("📄 Extracting text from document..."):
                text = analyzer.extract_text_from_pdf(uploaded_file)
            
            # Show debug information if enabled
            with st.expander("Debug Information", expanded=False):
                if st.checkbox("Show debug logs"):
                    st.text("Document Preview (first 500 chars):")
                    st.text(text[:500])
                    st.text("\nDocument Structure:")
                    st.text(f"Total length: {len(text)} characters")
                    paragraph_count = len([p for p in text.split('\n\n') if p.strip()])
                    st.text(f"Paragraphs: {paragraph_count}")
                    st.text(f"Selected scenarios: {scenarios}")
            
            # Analyze document for each selected scenario
            st.subheader("3. Analysis Results")
            
            with st.spinner("🔍 Analyzing document..."):
                all_findings = []
                
                # Progress bar for scenario analysis
                progress_bar = st.progress(0)
                total_scenarios = len(scenarios) + 1  # +1 for tax implications
                
                # Analyze main scenarios
                for i, scenario in enumerate(scenarios, 1):
                    st.write(f"Analyzing {scenario}...")
                    use_case = scenario.lower().replace(" ", "_")
                    context = {
                        "tenure": st.session_state.get('tenure', ''),
                        "leaving_type": st.session_state.get('leaving_type', ''),
                        "equity": st.session_state.get('equity', ''),
                        "joining_type": st.session_state.get('joining_type', '')
                    }
                    findings = analyzer.analyze_document(text, use_case, context=context)
                    if findings:
                        all_findings.extend([
                            (section, explanation, scenario, relevance_score)
                            for section, explanation, relevance_score in findings
                        ])
                    progress_bar.progress(i / total_scenarios)
                
                # Always analyze tax implications
                st.write("Analyzing tax implications...")
                tax_findings = analyzer.analyze_document(text, "tax_implications")
                if tax_findings:
                    all_findings.extend([
                        (section, explanation, "Tax Implications", relevance_score)
                        for section, explanation, relevance_score in tax_findings
                    ])
                progress_bar.progress(1.0)
            
            # Display results
            if all_findings:
                with st.spinner("📊 Organizing results..."):
                    # Add relevance filter
                    relevance_threshold = st.slider(
                        "Relevance Filter",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        help="Adjust to show more or fewer results based on relevance"
                    )
                    
                    # Group findings by scenario
                    for scenario in scenarios + ["Tax Implications"]:
                        scenario_findings = [f for f in all_findings if f[2] == scenario]
                        if scenario_findings:
                            # Sort by relevance score and filter
                            relevant_findings = sorted(
                                [f for f in scenario_findings if f[3] >= relevance_threshold],
                                key=lambda x: x[3],
                                reverse=True
                            )
                            
                            if relevant_findings:
                                st.markdown(f"### {scenario}")
                                
                                # Show summary of key findings
                                with st.expander("📋 Key Points Summary", expanded=True):
                                    st.markdown("**Most Important Considerations:**")
                                    for _, explanation, _, score in relevant_findings[:3]:
                                        st.markdown(f"- {explanation} *(Relevance: {score:.2f})*")
                                
                                # Show detailed findings
                                for i, (section, explanation, _, score) in enumerate(relevant_findings, 1):
                                    with st.expander(f"📑 Relevant Section {i} (Relevance: {score:.2f})"):
                                        col1, col2 = st.columns([3, 2])
                                        
                                        with col1:
                                            st.markdown("**📝 What we found:**")
                                            st.write(section)
                                        
                                        with col2:
                                            st.markdown("**❓ Why this matters:**")
                                            # Generate practical implications
                                            implications = generate_practical_implications(section, scenario, analyzer)
                                            
                                            # Key implications
                                            st.markdown("**Key Points:**")
                                            for point in implications["key_points"]:
                                                st.markdown(f"- {point}")
                                            
                                            # Practical actions
                                            st.markdown("**Actions to Consider:**")
                                            for action in implications["actions"]:
                                                st.markdown(f"- ✅ {action}")
                                            
                                            # Potential risks
                                            if implications["risks"]:
                                                st.markdown("**Risks to Consider:**")
                                                for risk in implications["risks"]:
                                                    st.markdown(f"- ⚠️ {risk}")
                with st.spinner("🎯 Generating practical analysis..."):
                    try:
                        scenario = scenarios[0].lower().replace(" ", "_") if scenarios else None
                        if scenario:
                            context = {
                                "tenure": st.session_state.get('tenure', ''),
                                "leaving_type": st.session_state.get('leaving_type', ''),
                                "equity": st.session_state.get('equity', ''),
                                "joining_type": st.session_state.get('joining_type', '')
                            }
                            
                            # Add debug logging
                            logger.info(f"Starting practical analysis for scenario: {scenario}")
                            logger.info(f"Context: {context}")
                            
                            try:
                                practical_analysis = analyzer.analyze_practical_implications(text, scenario, context)
                                logger.info(f"Practical analysis results: {practical_analysis}")
                                
                                if practical_analysis and any(practical_analysis.values()):
                                    display_practical_analysis(practical_analysis)
                                else:
                                    st.warning("No specific implications found. This might mean:")
                                    st.write("- The document doesn't contain explicit provisions for this scenario")
                                    st.write("- The relevant terms use different wording")
                                    st.write("- The scenario might be covered under general terms")
                            except Exception as e:
                                logger.error(f"Error in analyze_practical_implications: {str(e)}", exc_info=True)
                                st.error(f"Error analyzing implications: {str(e)}")
                                
                    except Exception as e:
                        logger.error(f"Error in practical analysis wrapper: {str(e)}", exc_info=True)
                        st.warning("Could not complete practical analysis. Please review the findings above.")
                
                # Show completion message
                st.success("✅ Analysis complete! Review the findings above.")
            else:
                st.warning("No relevant sections found in the document for the selected scenarios.")
                st.write("This might mean:")
                st.write("- The document doesn't explicitly address these scenarios")
                st.write("- The relevant terms use different wording")
                st.write("- The scenarios might be covered under general terms")
                
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            logging.error(f"Error processing document: {str(e)}", exc_info=True)

def display_practical_analysis(findings: Dict):
    """Display practical analysis results with context-specific calculations"""
    
    # Overview
    st.subheader("📋 Key Findings")
    for finding in findings["key_findings"]:
        with st.expander(f"🔍 {finding['element'].title()}"):
            st.markdown("**What we found:**")
            st.write(finding["content"])
            st.markdown("**Practical Impact:**")
            st.write(finding["practical_impact"])

    # Missing Elements
    if findings["missing_elements"]:
        st.warning("⚠️ Important Elements Not Found")
        st.write("The following important elements were not clearly addressed:")
        for element in findings["missing_elements"]:
            st.write(f"- {element}")
        st.write("Consider seeking clarification on these points.")

    # Action Items
    st.subheader("✅ Required Actions")
    for action in findings["action_items"]:
        st.write(f"- {action}")

    # Risks
    st.subheader("⚠️ Potential Risks")
    for risk in findings["risks"]:
        st.write(f"- {risk['description']} (Severity: {risk['severity']})")

    # Next Steps
    st.subheader("👉 Recommended Next Steps")
    for step in findings["next_steps"]:
        st.write(f"- {step}")

    # Display context-specific analysis if available
    if "context_specific_analysis" in findings and findings["context_specific_analysis"]:
        st.subheader("📊 Analysis Based on Your Specific Situation")
        
        context = findings["context_specific_analysis"]
        
        # Display equity calculations
        if "equity_calculation" in context:
            equity = context["equity_calculation"]
            with st.expander("🔍 Equity Status", expanded=True):
                if "error" in equity:
                    st.warning(equity["error"])
                else:
                    # Get vesting status safely
                    vesting_status = equity.get("vested_percentage", {}).get("vesting_status", {})
                    st.markdown(f"**Vesting Status:** {vesting_status.get('status', 'Not specified')}")
                    st.markdown(f"**Impact:** {vesting_status.get('impact', 'Not specified')}")
                    
                    # Show vesting percentage
                    vested = equity.get("vested_percentage", {})
                    if vested:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Vested Percentage", 
                                    f"{vested.get('vested_percent', 0)}%")
                        with col2:
                            st.metric("Forfeited", 
                                    f"{vested.get('forfeited_percent', 100)}%")
                        if "impact" in vested:
                            st.info(vested["impact"])
        
        # Display leaver status implications
        if "leaver_status" in context:
            leaver = context["leaver_status"]
            with st.expander("📑 Leaver Status Analysis", expanded=True):
                st.markdown(f"**Category:** {leaver['category']}")
                st.markdown("**Implications:**")
                for imp in leaver['implications']:
                    st.markdown(f"- {imp}")
                st.markdown(f"**Valuation Basis:** {leaver['valuation_basis']}")

def generate_practical_implications(section: str, scenario: str, analyzer: DocumentAnalyzer) -> Dict:
    """Generate practical implications using NLP and contextual analysis"""
    implications = {
        "key_points": [],
        "actions": [],
        "risks": []
    }
    
    # Get additional context if available
    if scenario == "founder_leaving":
        tenure = st.session_state.get('tenure', '')
        leaving_type = st.session_state.get('leaving_type', '')
        
        # Add context-specific implications
        if leaving_type == "Termination":
            implications["risks"].append("Review bad leaver provisions carefully")
            implications["actions"].append("Document compliance with notice periods")
        elif tenure and float(tenure.split()[0]) < 2:  # If less than 2 years
            implications["risks"].append("Early departure may affect vesting rights")
            
    elif scenario == "founder_joining":
        equity = st.session_state.get('equity', '')
        joining_type = st.session_state.get('joining_type', '')
        
        if joining_type == "Co-founder with limited rights":
            implications["actions"].append("Review voting rights limitations")
        if equity and float(equity.replace('%', '')) > 25:
            implications["actions"].append("Consider minority shareholder protections")
    
    # Use NLP to identify key elements
    doc = analyzer.nlp(section)
    
    # Extract key verbs and obligations
    obligations = [token.text for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"]
    subjects = [token.text for token in doc if "subj" in token.dep_]
    objects = [token.text for token in doc if "obj" in token.dep_]
    
    # Identify action triggers
    action_words = ["shall", "must", "will", "required", "agree", "undertake"]
    risk_words = ["unless", "except", "subject to", "provided that", "failure", "breach"]
    timing_words = ["within", "before", "after", "during", "upon"]
    
    # Generate key points based on sentence structure
    for sent in doc.sents:
        # Look for key obligations
        if any(word in sent.text.lower() for word in action_words):
            implications["key_points"].append(f"Requires action: {sent.text}")
            
        # Look for conditions and requirements
        if any(word in sent.text.lower() for word in risk_words):
            implications["risks"].append(f"Consider condition: {sent.text}")
            
        # Look for timing requirements
        if any(word in sent.text.lower() for word in timing_words):
            implications["actions"].append(f"Time-sensitive: {sent.text}")
    
    # Generate contextual actions based on scenario
    if scenario == "founder_leaving":
        if any(word in section.lower() for word in ["share", "equity", "stock"]):
            implications["actions"].append("Document current shareholding")
            implications["actions"].append("Review transfer restrictions")
    elif scenario == "founder_joining":
        if any(word in section.lower() for word in ["vest", "restriction", "right"]):
            implications["actions"].append("Review vesting conditions")
            implications["actions"].append("Understand share class rights")
    
    # Add general implications based on legal language
    if any(word in section.lower() for word in ["tax", "taxation", "hmrc"]):
        implications["actions"].append("Seek tax advice")
        implications["risks"].append("Tax implications need consideration")
    
    if any(word in section.lower() for word in ["consent", "approve", "permission"]):
        implications["actions"].append("Identify required approvals")
        implications["risks"].append("Failure to obtain necessary approvals")
    
    # Clean up and deduplicate
    for key in implications:
        implications[key] = list(set(implications[key]))
        implications[key] = [item for item in implications[key] if len(item) > 10]  # Remove very short items
    
    # Add default implications if none found
    if not any(implications.values()):
        implications["key_points"] = ["Review this section carefully"]
        implications["actions"] = ["Seek legal advice if unclear"]
        implications["risks"] = ["Ensure full understanding of obligations"]
    
    return implications

if __name__ == "__main__":
    main() 