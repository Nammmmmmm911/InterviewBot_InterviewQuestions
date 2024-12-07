import fitz  # PyMuPDF for PDF handling 
from transformers import AutoModelForCausalLM, AutoTokenizer
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

# Initialize ChromaDB Client
chroma_client = chromadb.Client()
collection_name = "company_data"
collection = chroma_client.get_or_create_collection(collection_name)

# Load GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

models = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Predefined list of job titles
job_titles = [
    "Software Engineer", "Data Scientist", "Cloud Engineer", "Full Stack Developer", 
    "DevOps Engineer", "Front End Developer", "Back End Developer", "Mobile Application Developer",
    "Cybersecurity Analyst", "Database Administrator", "System Administrator", "Network Engineer",
    "IT Support Specialist", "Web Developer", "Product Manager", "Machine Learning Engineer",
    "IT Project Manager", "Business Analyst", "Technical Support Engineer", 
    "Quality Assurance Engineer", "Data Engineer", "Artificial Intelligence Engineer", 
    "UX/UI Designer", "IT Consultant", "Solutions Architect", "IT Operations Manager", 
    "Chief Technology Officer (CTO)", "Security Engineer", "IT Auditor", "Software Architect",
    "Scrum Master", "Technical Writer", "Network Security Analyst", "Game Developer", 
    "Embedded Systems Engineer", "ERP Consultant", "Salesforce Developer", 
    "Big Data Engineer", "BI Developer", "Information Security Analyst", 
    "Robotics Engineer", "Cloud Solutions Architect", "Computer Vision Engineer", 
    "Site Reliability Engineer", "Penetration Tester", "Data Analyst", "Blockchain Developer",
    "IT Compliance Specialist", "Software Development Manager", "Virtual Reality Developer",
    "Infrastructure Engineer", "IT Operations Analyst", "Digital Marketing Specialist", 
    "Network Architect", "Help Desk Technician", "Configuration Manager", "Systems Analyst",
    "Database Developer", "IT Business Partner", "Cloud Consultant", "Virtualization Engineer",
    "E-commerce Specialist", "IT Trainer", "Technical Project Manager", "Mobile UX Designer",
    "Network Operations Center (NOC) Technician", "Release Manager", "IT Change Manager", 
    "Data Governance Analyst", "Performance Engineer", "BI Analyst", "SAP Consultant", 
    "Digital Transformation Consultant", "IT Asset Manager", "Game Designer", "Social Media Analyst"
]

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - str: Extracted text from the PDF.
    """
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def format_resume_text(resume_text):
    """
    Formats resume text to make main headings clear and visually appealing.
    
    Args:
    - resume_text (str): Raw text extracted from the resume.

    Returns:
    - str: Formatted text for better readability.
    """
    lines = resume_text.split("\n")
    formatted_lines = []

    for line in lines:
        line = line.strip()
        if line.isupper() and len(line.split()) < 6:
            formatted_lines.append(f"\n\n### {line} ###\n")
        elif line:  
            formatted_lines.append(line)

    return "\n".join(formatted_lines)

def extract_skills_using_ai(resume_text):
    """
    Extracts skills from the resume using the GPT-Neo model.

    Args:
    - resume_text (str): Full text of the resume.

    Returns:
    - str: Extracted skills as a string.
    """
    prompt = f"""
    You are a highly intelligent resume parser. Your task is to extract the skills section from the following resume text. 
    Return only the text below the 'Skills' section. If the 'Skills' section is not found, return an empty string.

    Resume Text:
    {resume_text}
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=600)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.strip()

def process_resume_and_match_jobs(pdf_file):
    """
    Processes the resume and matches it with job descriptions in ChromaDB.

    Args:
    - pdf_file: File-like object representing the uploaded PDF.

    Returns:
    - dict: Dictionary containing matched jobs.
    """
    # Step 1: Extract raw text from the resume
    resume_text = extract_text_from_pdf(pdf_file)

    # Step 2: Retrieve job descriptions from ChromaDB
    results = collection.get(include=["documents", "metadatas"])
    job_descriptions = results["documents"]
    job_titles = [metadata["jobTitle"] for metadata in results["metadatas"]]

    if not job_descriptions:
        return {
            "matched_jobs": ["No job descriptions available in ChromaDB."]
        }

    # Step 3: Match the resume text with job descriptions using TF-IDF and cosine similarity
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        all_documents = [resume_text] + job_descriptions
        tfidf_matrix = vectorizer.fit_transform(all_documents)

        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        top_indices = cosine_similarities.argsort()[-5:][::-1]  # Top 5 matches

        matched_jobs = [
            {"job_title": job_titles[i], "similarity": cosine_similarities[i]}
            for i in top_indices
        ]
    except ValueError as e:
        matched_jobs = [f"Error in processing job matching: {e}"]

    return {"matched_jobs": matched_jobs}

# Define the functions that generate questions for each role

def generate_software_engineer_questions():
    return [
        "What are the core principles of object-oriented programming (OOP)?",
        "Explain the concept of design patterns. Can you provide an example?",
        "Describe how you manage code versioning using Git.",
        "How do you ensure the quality and maintainability of your code?",
        "What testing strategies do you use during development?",
        "Can you explain the differences between functional and imperative programming?"
    ]

def generate_data_scientist_questions():
    return [
        "What data cleaning techniques do you typically use?",
        "How do you approach exploratory data analysis (EDA)?",
        "What is your experience with machine learning algorithms like regression or classification?",
        "Explain a time you optimized a model to improve accuracy.",
        "Which tools and frameworks do you prefer for data visualization?",
        "How do you deal with imbalanced datasets?"
    ]

def generate_cloud_engineer_questions():
    return [
        "What is your experience with cloud platforms like AWS, Azure, or Google Cloud?",
        "How do you ensure high availability and fault tolerance in the cloud?",
        "What is the difference between IaaS, PaaS, and SaaS?",
        "Explain how you approach security in cloud environments.",
        "What is your experience with containerization and orchestration tools like Docker and Kubernetes?",
        "How do you monitor cloud infrastructure and services?"
    ]

def generate_full_stack_developer_questions():
    return [
        "How do you ensure communication between the front-end and back-end of a full-stack application?",
        "What technologies do you use for building RESTful APIs?",
        "How would you handle user authentication and authorization in a web application?",
        "Explain the differences between SQL and NoSQL databases.",
        "How do you optimize the performance of both front-end and back-end systems?",
        "Can you walk us through your development process for a full-stack application?"
    ]

def generate_devops_engineer_questions():
    return [
        "What is your experience with continuous integration/continuous deployment (CI/CD)?",
        "How do you monitor system performance and ensure reliability?",
        "Can you explain the concept of infrastructure as code (IaC)?",
        "What tools do you use for automation and configuration management?",
        "How do you handle scaling in a cloud environment?",
        "Explain your approach to disaster recovery and business continuity."
    ]

def generate_front_end_developer_questions():
    return [
        "What is your experience with front-end frameworks like React, Angular, or Vue?",
        "How do you ensure responsive design across multiple devices?",
        "What is the difference between server-side rendering and client-side rendering?",
        "How do you optimize the performance of a front-end application?",
        "Can you explain the concept of state management in a front-end application?",
        "How do you handle cross-browser compatibility issues?"
    ]

def generate_back_end_developer_questions():
    return [
        "What is your experience with back-end technologies like Node.js, Python, or Ruby?",
        "How do you design scalable and maintainable APIs?",
        "Can you explain how you handle database migrations?",
        "What strategies do you use for error handling and logging?",
        "How do you ensure the security of your back-end systems?",
        "What is your approach to optimizing database queries for performance?"
    ]

def generate_mobile_application_developer_questions():
    return [
        "What is your experience with iOS and Android development?",
        "How do you manage app performance on different mobile devices?",
        "Can you explain the difference between native and hybrid mobile applications?",
        "What tools and frameworks do you use for mobile app testing?",
        "How do you handle offline functionality in mobile applications?",
        "What is your experience with mobile app security?"
    ]

def generate_cybersecurity_analyst_questions():
    return [
        "What is your experience with vulnerability assessments and penetration testing?",
        "How do you stay updated on the latest security threats and trends?",
        "Can you explain the difference between symmetric and asymmetric encryption?",
        "What strategies do you use to secure an organization's network?",
        "How do you approach incident response and handling a security breach?",
        "What tools do you use for network security monitoring?"
    ]

def generate_database_administrator_questions():
    return [
        "What is your experience with database optimization and indexing?",
        "How do you handle database backups and recovery?",
        "Explain your experience with database clustering and replication.",
        "What is your approach to data security and encryption in databases?",
        "How do you ensure database high availability and fault tolerance?",
        "Can you explain the differences between relational and non-relational databases?"
    ]

def generate_system_administrator_questions():
    return [
        "What is your experience with system monitoring and troubleshooting?",
        "How do you manage and configure servers for scalability?",
        "Can you explain the concept of virtualization and your experience with it?",
        "What tools do you use for automation and configuration management?",
        "How do you ensure the security of the systems you manage?",
        "Explain your experience with network configuration and firewall management."
    ]

def generate_network_engineer_questions():
    return [
        "What is your experience with network protocols like TCP/IP and DNS?",
        "How do you troubleshoot network connectivity issues?",
        "What is your experience with network performance monitoring tools?",
        "Can you explain how you would design a network infrastructure for a company?",
        "What strategies do you use to secure a network?",
        "How do you ensure high availability in network infrastructure?"
    ]

def generate_it_support_specialist_questions():
    return [
        "How do you prioritize and manage multiple support tickets?",
        "Can you explain how you diagnose hardware and software issues?",
        "What tools do you use for remote troubleshooting?",
        "How do you manage user permissions and access control?",
        "What is your approach to maintaining system documentation?",
        "How do you ensure a high level of customer satisfaction in support?"
    ]

def generate_web_developer_questions():
    return [
        "What are the key differences between HTML5 and its previous versions?",
        "How do you ensure that a website is responsive across different devices?",
        "Explain the box model in CSS and how it affects layout design.",
        "What is the difference between synchronous and asynchronous JavaScript?",
        "Describe how you would optimize a website for performance and SEO.",
        "What tools or frameworks do you use for front-end development?"
    ]

def generate_product_manager_questions():
    return [
        "What strategies do you use for gathering product requirements?",
        "How do you prioritize features in the product roadmap?",
        "Can you explain your experience with product lifecycle management?",
        "How do you collaborate with engineering, marketing, and design teams?",
        "What is your experience with agile methodologies in product development?",
        "How do you measure the success of a product after launch?"
    ]

def generate_machine_learning_engineer_questions():
    return [
        "What machine learning algorithms are you most comfortable with?",
        "How do you handle missing data in your models?",
        "What techniques do you use for feature selection?",
        "Explain a challenging machine learning project you worked on and your approach to solving it.",
        "What is your experience with deep learning frameworks like TensorFlow or PyTorch?",
        "How do you evaluate the performance of your machine learning models?"
    ]

# IT Project Manager
def generate_it_project_manager_questions():
    return [
        "Describe your approach to managing complex IT projects with tight deadlines.",
        "How do you ensure alignment between stakeholders during a project lifecycle?",
        "Explain how you manage risks and resolve conflicts in an IT project.",
        "Which project management tools do you prefer and why?",
        "How do you ensure your project team remains motivated and productive?",
        "Describe a challenging IT project you managed and how you ensured its success."
    ]


# Business Analyst
def generate_business_analyst_questions():
    return [
        "How do you identify and document business requirements effectively?",
        "What techniques do you use to perform gap analysis?",
        "How do you handle conflicting requirements from different stakeholders?",
        "Can you explain a time when you improved a business process significantly?",
        "Which software or tools do you prefer for creating workflows or wireframes?",
        "How do you ensure that your solutions align with organizational goals?"
    ]


# Technical Support Engineer
def generate_technical_support_engineer_questions():
    return [
        "How do you troubleshoot and resolve technical issues under pressure?",
        "Describe your experience with handling escalated support tickets.",
        "What tools or software do you prefer for remote troubleshooting?",
        "How do you communicate technical concepts to non-technical users?",
        "Explain a time when you solved a recurring issue in your support role.",
        "How do you stay updated on the latest technologies and troubleshooting techniques?"
    ]


# Quality Assurance Engineer
def generate_quality_assurance_engineer_questions():
    return [
        "How do you design effective test cases for complex applications?",
        "What strategies do you use for ensuring thorough regression testing?",
        "How do you prioritize bugs during the software testing process?",
        "What automation tools have you used, and how do you evaluate their effectiveness?",
        "Describe a time when you identified a critical bug late in the development cycle.",
        "How do you collaborate with developers to improve software quality?"
    ]


# Data Engineer
def generate_data_engineer_questions():
    return [
        "How do you design and optimize data pipelines for large-scale datasets?",
        "What experience do you have with distributed systems like Hadoop or Spark?",
        "How do you ensure data quality and integrity in your ETL processes?",
        "What tools or platforms do you use for real-time data processing?",
        "Explain how you have used cloud platforms for data engineering tasks.",
        "How do you manage and secure sensitive data in compliance with regulations?"
    ]


# Artificial Intelligence Engineer
def generate_ai_engineer_questions():
    return [
        "What is your experience with training and fine-tuning machine learning models?",
        "How do you optimize neural networks for performance and accuracy?",
        "Describe a project where you applied AI to solve a real-world problem.",
        "What frameworks and libraries do you prefer for AI development?",
        "How do you handle bias in AI models during development?",
        "What strategies do you use to deploy AI solutions into production?"
    ]


# UX/UI Designer
def generate_ux_ui_designer_questions():
    return [
        "How do you approach user research to inform your design process?",
        "What tools do you prefer for wireframing and prototyping?",
        "Describe how you have improved the usability of a product through design.",
        "How do you balance user needs with business goals in your designs?",
        "Explain how you collaborate with developers to implement your designs.",
        "What strategies do you use to gather and act on user feedback?"
    ]


# IT Consultant
def generate_it_consultant_questions():
    return [
        "How do you assess a company's IT infrastructure to recommend improvements?",
        "Describe a project where you implemented significant IT changes successfully.",
        "What methodologies do you use to ensure IT projects meet business goals?",
        "How do you stay updated with emerging technologies to advise clients?",
        "Explain a time when you had to manage resistance to IT changes.",
        "What tools do you use for conducting IT audits or assessments?"
    ]


# Solutions Architect
def generate_solutions_architect_questions():
    return [
        "How do you ensure a solution architecture meets both technical and business needs?",
        "Describe your approach to creating scalable and secure system designs.",
        "What cloud platforms do you have experience with, and how have you used them?",
        "How do you handle trade-offs between technical and cost considerations?",
        "Explain a time when you designed a solution that solved a critical problem.",
        "How do you work with stakeholders to validate solution requirements?"
    ]


# IT Operations Manager
def generate_it_operations_manager_questions():
    return [
        "How do you manage and optimize IT operations for maximum efficiency?",
        "What metrics do you track to evaluate IT operational performance?",
        "Describe your approach to managing incident response and minimizing downtime.",
        "How do you ensure compliance with IT policies and regulations?",
        "Explain a time when you successfully managed a major IT infrastructure upgrade.",
        "How do you prioritize IT investments to align with business goals?"
    ]


# Chief Technology Officer (CTO)
def generate_cto_questions():
    return [
        "How do you develop and communicate a clear technology vision for the company?",
        "What strategies do you use to align technology with business objectives?",
        "Describe a time when you led a major technology transformation in an organization.",
        "How do you manage competing priorities for technology investment?",
        "Explain your approach to building and managing a high-performing IT team.",
        "What emerging technologies do you believe will have the biggest impact in the future?"
    ]

# Security Engineer
def generate_security_engineer_questions():
    return [
        "How do you approach securing a network against potential threats?",
        "What tools and techniques do you use to conduct penetration testing?",
        "How do you stay updated with the latest cybersecurity vulnerabilities and threats?",
        "Describe a time when you responded to a security breach. What was your process?",
        "What experience do you have with implementing SIEM (Security Information and Event Management) systems?",
        "How do you ensure compliance with data security regulations such as GDPR or HIPAA?"
    ]


# IT Auditor
def generate_it_auditor_questions():
    return [
        "How do you evaluate an organization’s IT systems for compliance and security?",
        "What frameworks or standards do you use for IT audits (e.g., ISO 27001, COBIT)?",
        "Describe a time when you identified a significant risk during an audit. How did you address it?",
        "How do you prioritize tasks when auditing multiple systems or processes?",
        "What tools or software do you use to perform IT audits effectively?",
        "How do you communicate findings and recommendations to non-technical stakeholders?"
    ]


# Software Architect
def generate_software_architect_questions():
    return [
        "How do you design scalable and maintainable software architectures?",
        "What tools and techniques do you use to document system architectures?",
        "Describe your approach to selecting technologies for a new project.",
        "How do you ensure alignment between development teams and architectural goals?",
        "Explain a time when you had to refactor an existing system to improve performance or scalability.",
        "How do you manage technical debt in long-term projects?"
    ]


# Scrum Master
def generate_scrum_master_questions():
    return [
        "How do you ensure that your team adheres to Agile principles and Scrum practices?",
        "Describe your process for handling team conflicts during a sprint.",
        "What tools do you use to manage sprint planning and track progress?",
        "How do you measure and improve team velocity over time?",
        "Explain a time when you helped a team overcome obstacles to deliver a project on time.",
        "How do you communicate with stakeholders about project progress and risks?"
    ]


# Technical Writer
def generate_technical_writer_questions():
    return [
        "How do you create technical documentation that is accessible to various audiences?",
        "What tools or platforms do you use for writing and managing documentation?",
        "Describe your process for gathering information from subject matter experts.",
        "How do you ensure accuracy and clarity in technical documents?",
        "Can you provide an example of a complex topic you simplified through documentation?",
        "How do you stay updated on tools and trends in technical writing?"
    ]


# Network Security Analyst
def generate_network_security_analyst_questions():
    return [
        "How do you monitor and analyze network traffic for potential security threats?",
        "What tools do you use to detect and mitigate network vulnerabilities?",
        "Describe your process for responding to a DDoS (Distributed Denial of Service) attack.",
        "How do you implement and manage firewalls and intrusion detection systems?",
        "What strategies do you use to educate employees on network security best practices?",
        "How do you ensure compliance with network security standards and regulations?"
    ]


# Game Developer
def generate_game_developer_questions():
    return [
        "What game engines are you proficient in, and how have you used them in projects?",
        "Describe your process for optimizing game performance across platforms.",
        "How do you design engaging gameplay mechanics that enhance the user experience?",
        "What tools do you use for debugging and testing games during development?",
        "Explain how you work with artists and designers to implement game assets.",
        "Describe a challenging bug you encountered during development and how you resolved it."
    ]


# Embedded Systems Engineer
def generate_embedded_systems_engineer_questions():
    return [
        "What microcontrollers or processors have you worked with in your projects?",
        "How do you ensure real-time performance in embedded systems?",
        "Describe your experience with developing firmware for hardware devices.",
        "What debugging tools do you use for testing embedded systems?",
        "How do you optimize power consumption in low-power embedded devices?",
        "Explain a time when you designed a system with strict memory or resource constraints."
    ]


# ERP Consultant
def generate_erp_consultant_questions():
    return [
        "What ERP platforms have you worked with, and how have you implemented them?",
        "How do you ensure a smooth transition during an ERP system upgrade or migration?",
        "What strategies do you use to customize ERP solutions for specific business needs?",
        "Describe a challenging ERP implementation project and how you managed it.",
        "How do you train employees on using new ERP systems effectively?",
        "What tools do you use for integrating ERP systems with other enterprise applications?"
    ]


# Salesforce Developer
def generate_salesforce_developer_questions():
    return [
        "What experience do you have with developing custom Salesforce applications?",
        "How do you use Apex and Visualforce in your development process?",
        "Describe your approach to integrating Salesforce with external systems.",
        "What tools or strategies do you use to optimize Salesforce performance?",
        "How do you ensure data security and compliance in Salesforce applications?",
        "Explain a time when you customized Salesforce to meet a unique business requirement."
    ]


# Big Data Engineer
def generate_big_data_engineer_questions():
    return [
        "What tools and frameworks do you prefer for handling big data (e.g., Hadoop, Spark)?",
        "How do you design and maintain scalable big data pipelines?",
        "Describe your experience with managing structured and unstructured datasets.",
        "What strategies do you use to ensure data consistency and quality in big data systems?",
        "How have you implemented data security in big data environments?",
        "Explain a challenging big data project you worked on and how you delivered results."
    ]


# BI Developer
def generate_bi_developer_questions():
    return [
        "What BI tools and platforms are you proficient in (e.g., Tableau, Power BI)?",
        "How do you design dashboards that effectively communicate business insights?",
        "Describe your experience with writing complex SQL queries for data analysis.",
        "What strategies do you use to ensure accuracy and reliability in BI reports?",
        "Explain how you work with stakeholders to gather requirements for BI projects.",
        "Describe a time when your BI solution had a significant impact on business decisions."
    ]


# Information Security Analyst
def generate_information_security_analyst_questions():
    return [
        "How do you assess and mitigate security risks within an organization?",
        "What tools do you use for vulnerability scanning and threat detection?",
        "Describe your process for responding to and investigating security incidents.",
        "How do you ensure compliance with information security standards (e.g., ISO 27001)?",
        "Explain a time when you successfully prevented a security breach.",
        "What strategies do you use to educate employees on information security best practices?"
    ]

# Robotics Engineer
def generate_robotics_engineer_questions():
    return [
        "What programming languages and frameworks do you use for robotics development?",
        "Describe your experience with designing and implementing robotic control systems.",
        "How do you integrate sensors and actuators into robotic systems?",
        "What strategies do you use to optimize robot performance in dynamic environments?",
        "Describe a challenging robotics project you worked on and how you overcame obstacles.",
        "How do you approach testing and debugging in robotic systems?"
    ]


# Cloud Solutions Architect
def generate_cloud_solutions_architect_questions():
    return [
        "What experience do you have with designing cloud-based architectures for scalability and security?",
        "Describe your approach to choosing between different cloud service providers (e.g., AWS, Azure, GCP).",
        "How do you ensure compliance with data security and privacy regulations in cloud solutions?",
        "What tools and techniques do you use for cost optimization in cloud architectures?",
        "Explain a complex cloud migration project you worked on and how you managed it.",
        "How do you handle hybrid cloud or multi-cloud environments effectively?"
    ]


# Computer Vision Engineer
def generate_computer_vision_engineer_questions():
    return [
        "What tools and frameworks do you use for computer vision development (e.g., OpenCV, TensorFlow)?",
        "Describe your experience with designing and training deep learning models for vision tasks.",
        "How do you optimize computer vision algorithms for real-time performance?",
        "What strategies do you use to collect and preprocess data for computer vision projects?",
        "Explain a challenging computer vision problem you solved and the techniques you used.",
        "How do you ensure the robustness and accuracy of your computer vision models?"
    ]


# Site Reliability Engineer
def generate_site_reliability_engineer_questions():
    return [
        "How do you ensure high availability and reliability for distributed systems?",
        "What tools do you use for monitoring and automating infrastructure tasks?",
        "Describe your approach to incident response and root cause analysis.",
        "How do you manage and optimize CI/CD pipelines for development teams?",
        "What experience do you have with infrastructure as code (IaC) tools like Terraform or Ansible?",
        "Explain how you balance reliability with deployment velocity in a production environment."
    ]


# Penetration Tester
def generate_penetration_tester_questions():
    return [
        "What tools and techniques do you use to identify and exploit vulnerabilities in systems?",
        "Describe your process for performing a penetration test from start to finish.",
        "How do you report and communicate findings to stakeholders effectively?",
        "What strategies do you use to stay updated with the latest exploits and attack methods?",
        "Explain a challenging penetration test you conducted and the outcomes.",
        "How do you ensure compliance with ethical hacking and legal regulations?"
    ]


# Data Analyst
def generate_data_analyst_questions():
    return [
        "What tools and platforms do you use for data analysis (e.g., Excel, Python, Tableau)?",
        "Describe your experience with cleaning and preprocessing large datasets.",
        "How do you communicate your findings effectively to non-technical stakeholders?",
        "What techniques do you use for identifying trends and patterns in data?",
        "Explain a data analysis project where your insights had a significant impact.",
        "How do you ensure data accuracy and reliability in your reports?"
    ]


# Blockchain Developer
def generate_blockchain_developer_questions():
    return [
        "What blockchain platforms are you proficient in (e.g., Ethereum, Hyperledger)?",
        "Describe your experience with developing smart contracts using Solidity or other languages.",
        "How do you ensure the security of blockchain applications against potential vulnerabilities?",
        "What tools do you use for testing and debugging blockchain-based solutions?",
        "Explain a blockchain project you worked on and the challenges you faced.",
        "How do you optimize the performance of blockchain networks and applications?"
    ]


# IT Compliance Specialist
def generate_it_compliance_specialist_questions():
    return [
        "What frameworks or standards do you use to ensure IT compliance (e.g., GDPR, ISO 27001)?",
        "Describe your process for conducting IT compliance audits.",
        "How do you stay updated with changes in compliance regulations?",
        "What strategies do you use to train employees on IT compliance requirements?",
        "Explain a time when you identified and resolved a compliance issue.",
        "How do you collaborate with other departments to maintain IT compliance?"
    ]


# Software Development Manager
def generate_software_development_manager_questions():
    return [
        "How do you manage and prioritize tasks for development teams?",
        "Describe your approach to ensuring the quality and scalability of software projects.",
        "How do you handle conflicts or challenges within your development team?",
        "What strategies do you use to ensure on-time delivery of software projects?",
        "Explain a time when you implemented a new process to improve development efficiency.",
        "How do you balance technical leadership with project management responsibilities?"
    ]


# Virtual Reality Developer
def generate_virtual_reality_developer_questions():
    return [
        "What tools and frameworks do you use for VR development (e.g., Unity, Unreal Engine)?",
        "Describe your experience with designing immersive VR environments.",
        "How do you optimize VR applications for performance and user experience?",
        "What strategies do you use to integrate VR with other technologies (e.g., AR, AI)?",
        "Explain a VR project you worked on and the challenges you faced.",
        "How do you ensure accessibility and usability in VR applications?"
    ]

# Site Reliability Engineer
def generate_site_reliability_engineer_questions():
    return [
        "How do you monitor and maintain system reliability in a distributed environment?",
        "Describe your experience with incident response and root cause analysis.",
        "What tools and techniques do you use to automate infrastructure and deployment?",
        "How do you ensure high availability and fault tolerance in critical systems?",
        "Can you explain the concept of SLOs, SLIs, and SLAs and how you use them?",
        "What strategies do you use to scale systems to handle increased demand?"
    ]

# Infrastructure Engineer
def generate_infrastructure_engineer_questions():
    return [
        "What is your experience in designing and implementing IT infrastructure?",
        "How do you handle network bottlenecks and improve infrastructure performance?",
        "Describe a project where you successfully migrated infrastructure to the cloud.",
        "What tools do you use for infrastructure monitoring and management?",
        "How do you ensure security and compliance in IT infrastructure?",
        "Can you explain the process of capacity planning and resource allocation?"
    ]

# IT Operations Analyst
def generate_it_operations_analyst_questions():
    return [
        "How do you ensure smooth day-to-day IT operations?",
        "What tools do you use to monitor IT systems and prevent downtime?",
        "Describe your experience with ITIL or other service management frameworks.",
        "How do you handle escalations and prioritize incident response?",
        "What methods do you use to optimize IT operations and improve efficiency?",
        "Can you share an example where you implemented automation in IT operations?"
    ]

# Digital Marketing Specialist
def generate_digital_marketing_specialist_questions():
    return [
        "What is your experience with SEO and PPC campaigns?",
        "How do you analyze and improve the performance of digital marketing strategies?",
        "Which tools do you prefer for social media marketing and why?",
        "How do you measure ROI for digital marketing campaigns?",
        "Can you explain your approach to content marketing and brand awareness?",
        "Describe a successful campaign you managed and the results you achieved."
    ]

# Network Architect
def generate_network_architect_questions():
    return [
        "How do you design scalable and secure network architectures?",
        "What experience do you have with SDN (Software-Defined Networking)?",
        "Can you describe a time you optimized a network for performance and reliability?",
        "What tools and protocols do you use to monitor network traffic?",
        "How do you ensure network security against evolving threats?",
        "What is your process for evaluating and implementing new networking technologies?"
    ]

# Help Desk Technician
def generate_help_desk_technician_questions():
    return [
        "How do you handle troubleshooting technical issues for end-users?",
        "Describe a situation where you resolved a high-priority IT issue.",
        "What is your experience with ticketing systems like Jira or ServiceNow?",
        "How do you communicate technical solutions to non-technical users?",
        "What steps do you take to document recurring issues and their solutions?",
        "How do you stay updated with the latest IT support tools and techniques?"
    ]

# Configuration Manager
def generate_configuration_manager_questions():
    return [
        "How do you manage and maintain configuration baselines in IT systems?",
        "Describe your experience with version control systems like Git.",
        "What tools do you use for configuration management and automation?",
        "How do you ensure compliance with configuration standards and policies?",
        "Can you explain the importance of change management in configuration?",
        "Describe a scenario where you resolved a critical configuration issue."
    ]

# Systems Analyst
def generate_systems_analyst_questions():
    return [
        "How do you gather and analyze business requirements for system design?",
        "What tools and techniques do you use for process modeling and analysis?",
        "Describe your experience with systems integration and testing.",
        "How do you ensure the scalability and efficiency of system solutions?",
        "What methods do you use to stay updated on emerging technologies?",
        "Can you provide an example of a successful system implementation project?"
    ]

# Database Developer
def generate_database_developer_questions():
    return [
        "What is your experience in designing and optimizing database schemas?",
        "How do you ensure database performance and scalability?",
        "Describe your experience with stored procedures, triggers, and indexing.",
        "What tools do you use for database development and management?",
        "How do you handle database migrations and data integrity issues?",
        "Can you explain your approach to ensuring database security?"
    ]

# IT Business Partner
def generate_it_business_partner_questions():
    return [
        "How do you align IT strategies with business goals?",
        "Describe your experience collaborating with stakeholders to define IT priorities.",
        "What methods do you use to measure the ROI of IT initiatives?",
        "How do you identify opportunities for digital transformation in a business?",
        "What is your approach to managing IT budgets and resources?",
        "Can you share an example where you improved business outcomes through IT?"
    ]

# Cloud Consultant
def generate_cloud_consultant_questions():
    return [
        "What is your experience with cloud platforms like AWS, Azure, or GCP?",
        "How do you design cloud architectures to meet business requirements?",
        "Describe a successful cloud migration project you led or participated in.",
        "What tools and practices do you use for cloud cost optimization?",
        "How do you ensure security and compliance in cloud environments?",
        "Can you explain the advantages of serverless computing in the cloud?"
    ]

# Virtualization Engineer
def generate_virtualization_engineer_questions():
    return [
        "What is your experience with virtualization platforms like VMware or Hyper-V?",
        "How do you optimize virtualized environments for performance and efficiency?",
        "Describe your experience with virtual machine migrations and backups.",
        "What tools do you use for monitoring and managing virtualized systems?",
        "How do you ensure security in a virtualized infrastructure?",
        "Can you explain the benefits of containerization over traditional virtualization?"
    ]

# E-commerce Specialist
def generate_e_commerce_specialist_questions():
    return [
        "What is your experience with e-commerce platforms like Shopify or Magento?",
        "How do you optimize the user experience for e-commerce websites?",
        "Describe a successful strategy you used to increase online sales.",
        "What tools do you use for tracking and analyzing e-commerce metrics?",
        "How do you ensure security and compliance for online transactions?",
        "Can you share your approach to managing product catalogs and inventory online?"
    ]

# IT Trainer
def generate_it_trainer_questions():
    return [
        "What is your experience in developing and delivering IT training programs?",
        "How do you assess the effectiveness of your training sessions?",
        "Describe a time you trained a team on a new technology or tool.",
        "What methods do you use to engage learners during IT training?",
        "How do you customize training materials for different skill levels?",
        "What tools and platforms do you use for online IT training?"
    ]

# Technical Project Manager
def generate_technical_project_manager_questions():
    return [
        "What is your experience in managing technical projects from start to finish?",
        "How do you ensure projects stay on time and within budget?",
        "Describe your approach to risk management in technical projects.",
        "What tools do you use for project planning and tracking progress?",
        "How do you handle changes in project scope or requirements?",
        "Can you share an example of a challenging project you successfully delivered?"
    ]

# Mobile UX Designer
def generate_mobile_ux_designer_questions():
    return [
        "What is your process for designing user-friendly mobile interfaces?",
        "How do you conduct usability testing for mobile applications?",
        "Describe your experience with wireframing and prototyping tools.",
        "What are the key considerations for designing cross-platform mobile apps?",
        "How do you stay updated with trends in mobile UX design?",
        "Can you share an example of a successful mobile UX design project you worked on?"
    ]

# Network Operations Center (NOC) Technician
def generate_noc_technician_questions():
    return [
        "How do you monitor and troubleshoot network issues in real-time?",
        "What tools do you use for network performance monitoring?",
        "Describe your experience with handling network outages and escalations.",
        "How do you prioritize incidents in a high-pressure environment?",
        "What protocols and standards do you follow for network maintenance?",
        "Can you explain your process for documenting recurring network problems?"
    ]

# Release Manager
def generate_release_manager_questions():
    return [
        "What is your approach to coordinating software releases across teams?",
        "How do you ensure a smooth deployment process with minimal downtime?",
        "Describe your experience with version control and CI/CD tools.",
        "How do you manage and mitigate risks during a release?",
        "What is your process for tracking post-release issues and feedback?",
        "Can you provide an example of a challenging release you successfully managed?"
    ]

# IT Change Manager
def generate_it_change_manager_questions():
    return [
        "What is your process for managing IT changes in an organization?",
        "How do you communicate and coordinate with stakeholders during changes?",
        "Describe your experience with change management frameworks like ITIL.",
        "How do you assess the risks and impact of proposed changes?",
        "What tools do you use to track and document IT change requests?",
        "Can you share an example of a successful IT change implementation?"
    ]

# Data Governance Analyst
def generate_data_governance_analyst_questions():
    return [
        "What is your experience with establishing data governance policies?",
        "How do you ensure data compliance with regulations like GDPR or CCPA?",
        "Describe your process for managing data quality and integrity.",
        "What tools do you use for data lineage and cataloging?",
        "How do you collaborate with stakeholders to enforce data governance?",
        "Can you provide an example of improving data governance in a previous role?"
    ]

# Performance Engineer
def generate_performance_engineer_questions():
    return [
        "What tools do you use to identify and resolve system performance issues?",
        "Describe your experience with load testing and performance benchmarking.",
        "How do you optimize application performance under heavy loads?",
        "What metrics do you prioritize when evaluating system performance?",
        "How do you troubleshoot and address performance bottlenecks?",
        "Can you share an example of a successful performance optimization project?"
    ]

# BI Analyst
def generate_bi_analyst_questions():
    return [
        "What is your experience with BI tools like Tableau or Power BI?",
        "How do you design and build interactive dashboards for stakeholders?",
        "Describe your process for gathering and analyzing business requirements.",
        "What methods do you use to validate data accuracy and insights?",
        "How do you ensure BI solutions align with business goals?",
        "Can you provide an example of using BI to drive business decisions?"
    ]

# SAP Consultant
def generate_sap_consultant_questions():
    return [
        "What modules of SAP are you most experienced with?",
        "How do you gather requirements and customize SAP solutions for clients?",
        "Describe a successful SAP implementation project you worked on.",
        "What is your approach to troubleshooting SAP system issues?",
        "How do you ensure data migration and integration with SAP systems?",
        "Can you share your experience with SAP S/4HANA and its benefits?"
    ]

# Digital Transformation Consultant
def generate_digital_transformation_consultant_questions():
    return [
        "What is your approach to identifying areas for digital transformation?",
        "Describe your experience with implementing new technologies in organizations.",
        "How do you measure the success of digital transformation initiatives?",
        "What methods do you use to manage resistance to change during transformations?",
        "How do you ensure alignment between digital strategies and business goals?",
        "Can you provide an example of a successful digital transformation project?"
    ]

# IT Asset Manager
def generate_it_asset_manager_questions():
    return [
        "What is your process for tracking and managing IT assets?",
        "How do you ensure compliance with software licenses and regulations?",
        "Describe your experience with IT asset management tools.",
        "What methods do you use to optimize the lifecycle of IT assets?",
        "How do you handle asset disposals and data security during decommissioning?",
        "Can you share an example of improving efficiency through IT asset management?"
    ]

# Game Designer
def generate_game_designer_questions():
    return [
        "What is your process for creating engaging gameplay mechanics?",
        "How do you balance storytelling, design, and user experience in games?",
        "Describe your experience with game engines like Unity or Unreal Engine.",
        "What methods do you use to prototype and test game concepts?",
        "How do you collaborate with developers and artists during game production?",
        "Can you share an example of a successful game design project you worked on?"
    ]

# Social Media Analyst
def generate_social_media_analyst_questions():
    return [
        "What tools do you use to monitor and analyze social media performance?",
        "How do you measure the success of social media campaigns?",
        "Describe your approach to audience segmentation and targeting.",
        "How do you use social media analytics to drive engagement strategies?",
        "What methods do you use to track brand sentiment and competitor analysis?",
        "Can you share an example of improving ROI through social media insights?"
    ]

def generate_questions_for_job(job_title):

    questions = {
        "Software Engineer": generate_software_engineer_questions(),
        "Data Scientist": generate_data_scientist_questions(),
        "Cloud Engineer": generate_cloud_engineer_questions(),
        "Full Stack Developer": generate_full_stack_developer_questions(),
        "DevOps Engineer": generate_devops_engineer_questions(),
        "Front End Developer": generate_front_end_developer_questions(),
        "Back End Developer": generate_back_end_developer_questions(),
        "Mobile Application Developer": generate_mobile_application_developer_questions(),
        "Cybersecurity Analyst": generate_cybersecurity_analyst_questions(),
        "Database Administrator": generate_database_administrator_questions(),
        "System Administrator": generate_system_administrator_questions(),
        "Network Engineer": generate_network_engineer_questions(),
        "IT Support Specialist": generate_it_support_specialist_questions(),
        "Web Developer": generate_web_developer_questions(),
        "Product Manager": generate_product_manager_questions(),
        "Machine Learning Engineer": generate_machine_learning_engineer_questions(),
        "IT Project Manager": generate_it_project_manager_questions(),
        "Business Analyst": generate_business_analyst_questions(),
        "Technical Support Engineer": generate_technical_support_engineer_questions(),
        "Quality Assurance Engineer": generate_quality_assurance_engineer_questions(),
        "Data Engineer": generate_data_engineer_questions(),
        "Artificial Intelligence Engineer": generate_ai_engineer_questions(),
        "UI/UX Designer": generate_ux_ui_designer_questions(),
        "IT Consultant": generate_it_consultant_questions(),
        "Solutions Architect": generate_solutions_architect_questions(),
        "IT Operations Manager": generate_it_operations_manager_questions(),
        "Chief Technology Officer": generate_cto_questions(),
        "Security Engineer": generate_security_engineer_questions(),
        "IT Auditor": generate_it_auditor_questions(),
        "Software Architect": generate_software_architect_questions(),
        "Scrum Master": generate_scrum_master_questions(),
        "Technical Writer": generate_technical_writer_questions(),
        "Network Security Analyst": generate_network_security_analyst_questions(),
        "Game Developer": generate_game_developer_questions(),
        "Embedded Systems Engineer": generate_embedded_systems_engineer_questions(),
        "ERP Consultant": generate_erp_consultant_questions(),
        "Salesforce Developer": generate_salesforce_developer_questions(),
        "Big Data Engineer": generate_big_data_engineer_questions(),
        "BI Developer": generate_bi_developer_questions(),
        "Information Security Analyst": generate_information_security_analyst_questions(),
        "Robotics Engineer": generate_robotics_engineer_questions(),
        "Cloud Solutions Architect": generate_cloud_solutions_architect_questions(),
        "Computer Vision Engineer": generate_computer_vision_engineer_questions(),
        "Site Reliability Engineer": generate_site_reliability_engineer_questions(),
        "Penetration Tester": generate_penetration_tester_questions(),
        "Data Analyst": generate_data_analyst_questions(),
        "Blockchain Developer": generate_blockchain_developer_questions(),
        "IT Compliance Specialist": generate_it_compliance_specialist_questions(),
        "Software Development Manager": generate_software_development_manager_questions(),
        "Virtual Reality Developer": generate_virtual_reality_developer_questions(),
        "Infrastructure Engineer": generate_infrastructure_engineer_questions(),
        "IT Operations Analyst": generate_it_operations_analyst_questions(),
        "Digital Marketing Specialist": generate_digital_marketing_specialist_questions(),
        "Network Architect": generate_network_architect_questions(),
        "Help Desk Technician": generate_help_desk_technician_questions(),
        "Configuration Manager": generate_configuration_manager_questions(),
        "Systems Analyst": generate_systems_analyst_questions(),
        "Database Developer": generate_database_developer_questions(),
        "IT Business Partner": generate_it_business_partner_questions(),
        "Cloud Consultant": generate_cloud_consultant_questions(),
        "Virtualization Engineer": generate_virtualization_engineer_questions(),
        "E-commerce Specialist": generate_e_commerce_specialist_questions(),
        "IT Trainer": generate_it_trainer_questions(),
        "Technical Project Manager": generate_technical_project_manager_questions(),
        "Mobile UX Designer": generate_mobile_ux_designer_questions(),
        "Network Operations Center (NOC) Technician": generate_noc_technician_questions(),
        "Release Manager": generate_release_manager_questions(),
        "IT Change Manager": generate_it_change_manager_questions(),
        "Data Governance Analyst": generate_data_governance_analyst_questions(),
        "Performance Engineer": generate_performance_engineer_questions(),
        "BI Analyst": generate_bi_analyst_questions(),
        "SAP Consultant": generate_sap_consultant_questions(),
        "Digital Transformation Consultant": generate_digital_transformation_consultant_questions(),
        "IT Asset Manager": generate_it_asset_manager_questions(),
        "Game Designer": generate_game_designer_questions(),
        "Social Media Analyst": generate_social_media_analyst_questions()
    }
     # Normalize job title to avoid case or whitespace issues
    job_title = job_title.strip()  # Remove leading/trailing spaces
    if job_title in questions:
        # Debugging log
        print(f"Found job title: {job_title}")
        return questions[job_title]
    else:
        # Debugging log
        print(f"Job title not found: {job_title}. Returning default questions.")
        return [
            f"What relevant experience do you have for the role of {job_title}?",
            f"What technologies have you worked with that are essential for {job_title}?",
            f"Describe a challenge you faced in a similar role to {job_title} and how you resolved it.",
            f"How would you approach a project in the position of {job_title}?"
        ]
    
def provide_feedback(answers):
    """
    Processes the user's answers to generate a compatibility score and feedback.
    
    Args:
        answers (list of str): The answers submitted by the user.
    
    Returns:
        tuple: A compatibility score (int) and feedback (str).
    """
    # Example scoring logic: Count the number of meaningful answers
    # Here, we're simply checking if the answer contains some expected content.
    score = 0
    for answer in answers:
        if len(answer.strip()) > 20:  # A valid answer must be at least 20 characters
            score += 1

    # Generate feedback based on the score
    if score >= len(answers) * 0.8:
        feedback = "Excellent! You have strong responses for this role."
    elif score >= len(answers) * 0.5:
        feedback = "Good! But there's room for improvement in some answers."
    else:
        feedback = "Needs improvement. Consider revising your answers."

    return score, feedback






