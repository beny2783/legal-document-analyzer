import streamlit as st
from document_analyzer import DocumentAnalyzer
import logging
from typing import Dict

def main():
    st.title("Legal Document Analyzer")
    st.write("Analyze legal documents for specific scenarios")

    # Initialize the analyzer
    analyzer = DocumentAnalyzer()

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
                    findings = analyzer.analyze_document(text, use_case)
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
                                    with st.expander(
                                        f"📑 Relevant Section {i} (Relevance: {score:.2f})"
                                    ):
                                        st.markdown("**What we found:**")
                                        st.write(section)
                                        st.markdown("**Why this matters:**")
                                        st.write(explanation)
                with st.spinner("🎯 Generating practical analysis..."):
                    try:
                        scenario = scenarios[0].lower().replace(" ", "_") if scenarios else None
                        if scenario:
                            practical_analysis = analyzer.analyze_practical_implications(text, scenario)
                            if practical_analysis:
                                display_practical_analysis(practical_analysis)
                    except Exception as e:
                        logger.error(f"Error in practical analysis: {str(e)}")
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
    """Display practical analysis results"""
    
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

if __name__ == "__main__":
    main() 