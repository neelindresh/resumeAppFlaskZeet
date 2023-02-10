from flask import Markup
RESUME_LINK="pdf/resume2023.pdf"

ABOUT_ME={
    "position":"Data Scientist | Author PyMLPipe",
    "about": "I have extensive experience delivering end-to-end analytics solutions, having successfully completed over 15 projects across various industries. My expertise lies in leveraging data to uncover insights, drive business outcomes, and create value for organizations. I am a strategic thinker with excellent problem-solving skills, and I have a proven track record of delivering innovative solutions that meet and exceed expectations. I am eager to bring my expertise and passion for data science to a dynamic and forward-thinking organization.",
    "intro": "A self-motivated, tech-savvy and result oriented professional who always looks for opportunity to exercise creative and innovative ideas to make contribution to the development of the organization. Experienced in creating end-to- end Analytics solutions.",
    
}

ABOUT_DETAILS=[
    {"exp":"05+ Years", "icon":"uil uil-briefcase-alt","title":"Experience"},
    {"exp":"15+ Completed",'icon':"uil uil-apps","title":"Projects"},
    {"exp":"10+ Client",'icon':"uil uil-constructor","title":"Happy"}
]

SKILLS=[
    {
        "header":"Programming",
        'target':"coding",
        "icon":"uil uil-code-branch",
        "exp":"05+",
        "tools":[
            {
                    "name":"Python",
                    "perct":90,
            },
            {
                    "name":"C",
                    "perct":60,
            },
            {
                    "name":"C++",
                    "perct":60,
            },
            {
                    "name":"Java",
                    "perct":50,
            },
            
        ]
    },
    {
        "header":"Machine Learning",
        'target':"ml",
        'icon':"uil uil-robot",
        "exp":"05+",
        "tools":[
            {
                    "name":"Scikit-learn",
                    "perct":90,
            },
            {
                    "name":"Pandas",
                    "perct":70,
            },
            {
                    "name":"Numpy",
                    "perct":70,
            },
            {
                    "name":"Predictive Modeling",
                    "perct":80,
            },
            {
                    "name":"Statistical Modeling",
                    "perct":80,
            },
            {
                    "name":"Forcasting",
                    "perct":60,
            },
            {
                    "name":"Supervised Learning",
                    "perct":90,
            },
            {
                    "name":"Unsupervised Learning",
                    "perct":80,
            },
        ]
    },
    {
        "header":"Deep Learning",
        'target':"dl",
        "icon":"uil uil-channel-add",
        "exp":"05+",
        "tools":[
            {
                    "name":"Pytorch",
                    "perct":90,
            },
            {
                    "name":"Tensorflow",
                    "perct":50,
            },
            {
                    "name":"Computer Vision (CV)",
                    "perct":70,
            },
            {
                    "name":"Natural Language Processing(NLP)",
                    "perct":80,
            },
            {
                    "name":"NLU",
                    "perct":80,
            },
            {
                    "name":"Generative Modeling ",
                    "perct":60,
            },
            
        ]
    },
    {
        "header":"Data Orchestration",
        'target':"cloud",
        "icon":"uil uil-cloud-computing",
        "exp":"05+",
        "tools":[
            {
                    "name":"Airflow",
                    "perct":70,
            },
            {
                    "name":"Docker",
                    "perct":80,
            },
            {
                    "name":"PySpark",
                    "perct":60,
            },
            {
                    "name":"MLOps",
                    "perct":90,
            },
            {
                    "name":"NoSQL",
                    "perct":80,
            },
            {
                    "name":"SQL",
                    "perct":60,
            },
            
        ]
    }
    
]

QUALIFICATION={
        "experience":[
                        {
                                "organization":"KPMG India",
                                "title":"Consultant",
                                "years":"2021-present"
                        },
                        {
                                "organization":"PRM Fincon",
                                "title":"Data Scientist",
                                "years":"2020-2021"
                        },
                        {
                                "organization":"Virtusa",
                                "title":"ASSOCIATE ML ENGINEER | ML ENGINEER",
                                "years":"2018-2020"
                        }
                ],
        "education":[
                        {
                                "organization":"Vellore Institute of Technology(VIT)",
                                "title":"Master of Compter Application: Data Science",
                                "years":"2016-2018"
                        },
                        {
                                "organization":"West Bengal State university",
                                "title":"Bachelor of Science: Computer Science",
                                "years":"2013-2016"
                        },
                        
                ],
}


    
PROJECTS=[
        {
                "title":"PyMlPipe | MLOps toolbox",
                "desc":"Pymlpipe is a cutting-edge open-source machine learning operations (MLOps) toolkit built in Python that provides an all-in-one solution for managing and streamlining machine learning projects. It includes features such as model monitoring, data monitoring, infrastructure monitoring, and data drift detection. It also integrates with other popular open-source tools such as DVC, Concept Drift Detection, and eXplainable Artificial Intelligence (XAI). Pymlpipe is designed to be user-friendly and accessible, making it a great choice for organizations and individuals looking for a comprehensive MLOps solution. It is well-positioned to help users stay ahead of the curve and stay competitive in today's rapidly-evolving machine learning landscape.",
                "keypoints":[],
                "role":"Data Scientist",
                "org": "Open-Source",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Compter Vision","MLOps","DVC"],
                "Links":"https://github.com/neelindresh/pymlpipe",
                "category":"MLOps",
                "year": 2023
        },
        {
                "title":"Finding writing tools for Auditors",
                "desc":'''A Writing Tool for Auditors is a software application that helps auditors in their report writing process by providing features such as grammatical error correction, text scoring, text rephrasing, and segregating findings description into issue impact, background, and observation. It aims to enhance the efficiency, accuracy, and consistency of report writing for auditors by automating certain manual tasks and reducing the likelihood of human error. The tool can be built using various technologies such as NLP, AI, and machine learning. It is designed to define the requirements and feature set of the tool, research and analyze existing solutions in the market, and design the architecture of the tool. It also integrates all the modules and test the tool for bugs and performance. Finally, it releases the tool and continually improves it based on user feedback..''',
                "keypoints":['Research: Start by researching the available tools in the market and the features they provide. This will give you a good idea of what is possible and what is not.', 'Determine the Requirements: Identify the key requirements for the tool, such as grammatical error correction, text scoring, text rephrasing, and the ability to segregate findings descriptions into issue impact, background, and observation.', 'Design the Architecture: Design the overall architecture of the tool, keeping in mind the features you want to include. Consider using an NLP library or toolkit to handle the text processing, scoring, and rephrasing features.', 'Implementation: Start coding the tool based on the architecture design. You can use Python or other programming languages to build the tool.', 'Testing and Debugging: Test the tool to make sure it works as expected. Debug any issues that arise and make any necessary changes.', 'Deployment: Deploy the tool and make it available to auditors. You can either deploy it as a standalone tool or integrate it into an existing audit management system.', 'Maintenance and Support: Provide ongoing maintenance and support for the tool to ensure its continued performance and reliability.'],
                "role":"Data Scientist",
                "org": "KPMG",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Transformer","API","spaCy", "NLTK", "Gensim"],
                "Links":"",
                "category":"DL",
                "year": 2022
        },
        {
                "title":"Iron Manufacturing Maintenance and Sustenance Prediction",
                "desc":"The Iron Manufacturing Maintenance and Sustenance Prediction project is aimed at developing a predictive model that can predict the maintenance requirements and overall sustenance of iron manufacturing units. The model would be built using historical data related to the manufacturing process and maintenance records, along with relevant features such as temperature, pressure, and raw material quality. Advanced techniques like machine learning and artificial intelligence can be used to develop the predictive model. The objective of this project is to increase the efficiency and reduce the costs associated with the maintenance of iron manufacturing units by anticipating potential issues before they occur. The output of this project would be a tool or software application that can be used by iron manufacturing companies to optimize their maintenance operations and improve the overall sustainability of their operations.",
                "keypoints":['Data collection and pre-processing: Collect and clean the data related to iron manufacturing, including machine operations, maintenance records, and other relevant factors.', 'Feature engineering: Identify the relevant features and create new ones to best represent the data for modeling.', 'Model selection: Select the appropriate machine learning model based on the type of problem, data, and desired outcomes.', 'Training: Train the model using the pre-processed data, ensuring that the model can make accurate predictions.', 'Validation: Evaluate the performance of the model by comparing the predictions to actual results, and make any necessary adjustments.', 'Deployment: Deploy the model in a real-time environment and integrate it into the iron manufacturing maintenance system to support predictive maintenance and decision making.', 'Monitoring and evaluation: Monitor the performance of the tool and collect feedback to continuously improve it.'],
                "role":"Data Scientist",
                "org": "KPMG",
                "tech": ["python","scikit-learn","Machine Learning","API","Flask","MLOps","mlflow","pandas"],
                "Links":"",
                "category":"ML",
                "year": 2022
        },
        {
                "title":"Metal & Mining Predictive maintenance",
                "desc":"Microstructure and Mechanical Property Prediction is a project that involves using machine learning algorithms to predict the microstructure and mechanical properties of materials based on various input parameters. The goal of this project is to build a model that can accurately predict the properties of materials based on their chemical composition, processing conditions, and other relevant factors. This can help reduce the cost and time required to perform physical testing of materials, and can also provide more insights into the underlying relationships between microstructure and mechanical properties. ",
                "keypoints":['Data Collection: Obtain a dataset of microstructure and mechanical property data, typically from experiments or simulations.', 'Data Preprocessing: Clean and preprocess the data, handling any missing or inconsistent values, and normalizing the data as needed.', 'Model Selection: Choose a machine learning model that is appropriate for the task at hand. This might involve comparing the performance of different models on a subset of the data.', "Model Training: Train the chosen model on the preprocessed data. This involves optimizing the model's parameters so that it can accurately predict the target mechanical property from the microstructure data.", 'Model Evaluation: Evaluate the performance of the trained model on a validation dataset. This might involve calculating metrics such as accuracy, precision, recall, or F1 score.', 'Model Deployment: Deploy the trained model as a tool that can be used to make predictions on new microstructure data.'],
                "role":"Data Scientist",
                "org": "KPMG",
                "tech": ["python","scikit-learn","Machine Learning","API","Flask","MLOps","mlflow","pandas"],
                "Links":"",
                "category":"ML",
                "year": 2022
        },
         {
                "title":"Named Entity Recognition for Financial forms",
                "desc":"Named Entity Recognition (NER) is a task in Natural Language Processing (NLP) that involves identifying and classifying named entities such as person names, location names, organization names, and more, in a given text. The project focuses on using Optical Character Recognition (OCR) to extract text from financial forms and then using NER to identify named entities in the extracted text. The goal is to provide auditors with a tool that can help them process large volumes of financial forms automatically and extract relevant information efficiently, saving time and effort. The project will likely involve the use of NLP libraries and OCR tools, as well as the development of a machine learning model trained on financial data to perform NER on financial forms. Advantages:- Automated Data Extraction,Improved Accuracy,Reduced Time,Improved Consistency,Enhanced Analytics",
                "keypoints":['Data Collection: Collect a large number of financial forms for training the NER model.', 'Image Pre-processing: Clean and pre-process the images of financial forms to remove any noise and improve the quality of the image for OCR.', 'Text Extraction: Use OCR technology to extract text from images of financial forms.', 'Data Preparation: Prepare the extracted text data for NER by splitting it into sentences and tokens.', 'NER Model Training: Train a NER model on the prepared data, using Deep Learning algorithms like LSTMs or transformers.', 'NER Model Testing: Test the model on a separate set of financial forms to evaluate its performance.', 'Deployment: Deploy the NER model in a system that can automatically recognize named entities in financial forms using OCR technology.'],
                "role":"Data Scientist",
                "org": "PRM Fincon",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Transformer","API","spaCy", "NLTK", "Gensim"],
                "Links":"",
                "category":"ML",
                "year": 2021
        },
        {
                "title":"Distributed Architecture Framework",
                "desc":"A Distributed Architecture Framework built using core python as an alternative to Apache Airflow can help organizations in designing and managing complex workflows efficiently. This framework enables the user to run tasks and dependencies across multiple systems and data sources in a scalable and resilient manner. The framework will provide a simple and efficient way to manage tasks and dependencies, as well as monitor and manage the execution of these tasks. By using core python, this framework can provide a more straightforward and less complex solution compared to Apache Airflow while still delivering the desired functionality.",
                "keypoints":['Define the problem statement and requirements for the framework.', 'Choose the appropriate distributed architecture pattern that fits the problem statement and requirements, such as Master-Worker or Map-Reduce.', 'Design the architecture and components of the framework, including task scheduler, task executor, data storage, and communication mechanism.', 'Implement the framework using core python programming language, leveraging libraries such as multiprocessing and queue to handle parallelism and task management.', 'Test the framework thoroughly to ensure its performance, scalability, and reliability.', 'Integrate the framework with existing systems and tools, such as database systems and data pipelines.', 'Monitor the performance and make necessary optimizations to improve the efficiency of the framework.'],
                "role":"Data Scientist",
                "org": "PRM Fincon",
                "tech": ["python","API","Distributed systems","RPC","Scheduler"],
                "Links":"",
                "category":"Others",
                "year": 2020
        },
         {
                "title":"Financial document Classification",
                "desc":"Financial document classification is the process of automatically categorizing financial documents into different classes or categories based on their contents. It can be performed on various financial documents such as balance sheets, income statements, invoices, receipts, etc. and can be used for various tasks such as financial statement analysis, fraud detection, and regulatory compliance. AI can help organizations organize their financial data, reduce manual effort, and improve efficiency. It can also help auditors identify and organize relevant documents for the audit process, saving time and effort, and reducing the risk of errors.",
                "keypoints":['Data Collection: Gather a diverse set of financial documents that need to be classified and organize them into different categories or classes.', 'Data Pre-processing: Clean and pre-process the data to remove any irrelevant information or formatting issues.', 'Feature Engineering: Extract relevant features from the financial documents that can be used as inputs for the model.', 'Model Selection: Choose an appropriate machine learning model for the classification task.', "Model Training: Train the model on the pre-processed data and fine-tune the model's hyperparameters to optimize its performance.", 'Model Evaluation: Evaluate the performance of the model using appropriate metrics and make any necessary changes to improve its accuracy.', 'Deployment: Deploy the model into a production environment and integrate it with any necessary tools or platforms.'],
                "role":"Data Scientist",
                "org": "PRM Fincon",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Computer Vision","OCR","API"],
                "Links":"",
                "category":"DL",
                "year": 2020
        },
        {
                "title":"Financial document extraction",
                "desc":"Financial document extraction is a process that uses AI and NLP techniques to automatically extract relevant information from financial documents. Machine learning algorithms, natural language processing (NLP) techniques, and computer vision may be used to build the model. Once the model has been developed, it can be integrated into a financial document management system or accounting software to automate the process of extracting financial information. AI can greatly assist auditors in the review and analysis of financial records, saving time and increasing accuracy. OCR and NLP can also be used to perform financial document extraction. This technology can provide valuable support for auditors in their efforts to ensure the accuracy and integrity of financial records.",
                "keypoints":['Data Collection: To build an effective model, a large and diverse dataset of financial documents is required. This dataset will be used to train the model on how to identify and extract relevant information from financial documents.', "Data Cleaning and Preprocessing: The collected data may contain noise, irrelevant information, and inconsistent data structures. To improve the model's accuracy, the data must be cleaned and preprocessed to standardize the data format and remove any irrelevant information.", 'Feature Engineering: Feature engineering is the process of creating features that the model can use to identify and extract relevant information from financial documents. This may include creating features such as keywords, phrases, entity recognition, and image recognition features.', 'Model Development: Based on the preprocessed data, machine learning algorithms and NLP techniques are used to develop a model that can automatically extract relevant information from financial documents. This may involve using deep learning algorithms, such as convolutional neural networks (CNNs) or recurrent neural networks (RNNs), or traditional machine learning algorithms such as decision trees or support vector machines (SVMs).', 'Model Validation and Testing: To ensure that the model is accurate and reliable, it must be tested on a separate dataset of financial documents. This will help to identify any errors in the model and ensure that it is working as expected.', 'Integration and Deployment: Once the model has been validated, it can be integrated into a financial document management system or accounting software to automate the process of extracting financial information from financial documents. The model must also be deployed in a scalable and secure environment, to ensure that it can handle the volume of financial documents that it will be processing.'],
                "role":"Data Scientist",
                "org": "PRM Fincon",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Computer Vision","OCR","API"],
                "Links":"",
                "category":"DL",
                "year": 2020
        },
        {
                "title":"Automatic Financial document tagging",
                "desc":"This project involves developing a model that can automatically identify and extract key information from financial documents. Machine learning algorithms, natural language processing (NLP) techniques and computer vision may be used to build the model. Once the model has been developed, it can be integrated into a larger financial document management system. The goal is to make the process of organizing financial documents more efficient and accurate. For example, NLP techniques can be used to identify and extract relevant information from the text of the financial documents, while computer vision can be used to extract information from any images or tables in the document.  ",
                "keypoints":['Data Collection: The first step is to collect a dataset of images, videos, and text-based content that will be used to train and test the content classification model. This dataset should include examples of both appropriate and inappropriate content.', 'Data Pre-processing: Before building the model, the dataset needs to be pre-processed to remove any irrelevant or redundant information, and to format the data in a way that is suitable for use with machine learning algorithms.', 'Feature Engineering: Next, the relevant features of the content need to be extracted and represented in a form that can be used by the model. This may involve applying computer vision techniques to extract features from images and videos, or NLP techniques to extract features from text-based content.', 'Model Training: The next step is to train a deep learning model using the pre-processed and feature-engineered dataset. This typically involves using algorithms such as convolutional neural networks (CNNs) for image classification or recurrent neural networks (RNNs) for text classification.', "Model Validation: After training the model, it is important to evaluate its performance to ensure that it is accurate and reliable. This can be done by using a validation dataset and comparing the model's predictions with the ground truth labels.", 'Deployment: Finally, once the model has been validated and shown to be accurate, it can be deployed in production to automatically classify new content as appropriate or not appropriate, based on its content.'],
                "role":"Data Scientist",
                "org": "PRM Fincon",
                "tech": ["python","PyTorch","Deep Learning" ,"NLP","Computer Vision",],
                "Links":"",
                "category":"DL",
                "year": 2020
        },
        {
                "title":"Google Ads Validation framework",
                "desc":"Classifying not family-friendly content in Google Ads uses NLP (Natural Language Processing) and computer vision. The goal is to ensure that all content displayed is safe and suitable for the intended audience. NLP techniques can be used to analyze text-based content, such as ad descriptions, to identify words and phrases that are commonly associated with inappropriate content. Computer vision tools can be combined to analyze images and videos, identifying objects and scenes that are inappropriate.",
                "keypoints":['Data Collection: The first step is to collect a dataset of images, videos, and text-based content that will be used to train and test the content classification model. This dataset should include examples of both appropriate and inappropriate content.', 'Data Pre-processing: Before building the model, the dataset needs to be pre-processed to remove any irrelevant or redundant information, and to format the data in a way that is suitable for use with machine learning algorithms.', 'Feature Engineering: Next, the relevant features of the content need to be extracted and represented in a form that can be used by the model. This may involve applying computer vision techniques to extract features from images and videos, or NLP techniques to extract features from text-based content.', 'Model Training: The next step is to train a deep learning model using the pre-processed and feature-engineered dataset. This typically involves using algorithms such as convolutional neural networks (CNNs) for image classification or recurrent neural networks (RNNs) for text classification.', "Model Validation: After training the model, it is important to evaluate its performance to ensure that it is accurate and reliable. This can be done by using a validation dataset and comparing the model's predictions with the ground truth labels.", 'Deployment: Finally, once the model has been validated and shown to be accurate, it can be deployed in production to automatically classify new content as appropriate or not appropriate, based on its content.'],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","PyTorch","Deep Learning" ,"spaCy", "NLTK", "Gensim","Machine Learning","NLP","Computer Vision"],
                "Links":"",
                "category":"DL",
                "year":2019
        },
        {
                "title":"GEC Google Ads",
                "desc":"A grammar error correction platform using deep learning for writing Google Ads is a specialized system designed to help marketers and advertisers improve the quality and accuracy of their written ad copy specifically for Google Ads. This platform leverages deep learning algorithms to detect and correct grammatical errors in written text, with a focus on optimizing ad copy for Google's ad platform. The platform uses advanced NLP techniques, such as recurrent neural networks (RNNs) and transformer models, to process and analyze written text, identify grammatical errors, and suggest corrections that are tailored to meet Google's ad copy standards. This specialized grammar error correction platform can help advertisers create more effective and efficient ad copy, ultimately improving the performance and return on investment of their Google Ads campaigns.",
                "keypoints":['Data Collection: The first step is to gather and pre-process a large dataset of text that contains grammatical errors, along with their corrected versions. This dataset will be used to train the deep learning model.', 'Data Pre-processing: The collected text data needs to be cleaned and pre-processed to prepare it for use in the deep learning model. This step includes tasks such as removing special characters, converting text to lowercase, and tokenizing the text into words and sentences.', 'Model Design: Next, design the deep learning model architecture. This can include choosing the type of deep learning algorithms such as RNNs, LSTM networks, or transformer models, determining the number of hidden layers, and defining the size of the input layer.', 'Model Training: Train the deep learning model on the pre-processed data. This involves feeding the model the text data, along with the corrected versions, and updating the model weights and biases to minimize the error between the model predictions and the actual corrected text.', 'Model Evaluation: Evaluate the performance of the deep learning model using accuracy metrics such as precision, recall, and F1 score. This step is also used to identify and address overfitting, which is when the model has learned to make predictions based on the training data, but has poor generalization to new, unseen data.', 'Model Deployment: Once the model is trained and evaluated, it can be deployed to the platform and integrated with Google Ads.', 'Continuous Improvement: Finally, monitor the performance of the model over time and make improvements as necessary. This can include retraining the model on new data, fine-tuning hyperparameters, or updating the model architecture.'],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","PyTorch","Deep Learning","scikit-learn" ,"spaCy", "NLTK", "Gensim","NLP","RNNs","LSTM ","Transformer"],
                "Links":"",
                "category":"DL",
                "year":2019
        },
        {
                "title":"DQ2, Data Quality Platform",
                "desc":"A data quality platform is a system designed to monitor, analyze, and improve the quality of data. It typically includes features such as data cleansing, data enrichment, data matching, and data reconciliation to improve the accuracy and completeness of the data. It can be critical for organizations that rely on data for decision making, as poor quality data can lead to incorrect insights and decisions. AI-powered data quality platforms can handle large volumes of data in real-time, automate manual and time-consuming tasks, and make the process more efficient and effective. These platforms are becoming increasingly important for organizations relying on data for critical business decisions, as they help to ensure that the data is accurate, consistent, and reliable.",
                "keypoints":[],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","PyTorch","scikit-learn" ,"spaCy", "NLTK", "Gensim","Machine Learning","NLP"],
                "Links":"",
                "category":"ML",
                "year":2019
        },
        {
               "title":"AI-celerate, GUI based AI platform",
                "desc":"An AI platform is a comprehensive system that enables organizations to develop, deploy, and manage artificial intelligence models and applications. It provides an end-to-end solution for building, training, and deploying machine learning models, and it often includes features such as data management, model development tools, and deployment infrastructure. The goal of an AI platform is to make it easier for organizations to incorporate AI into their operations and workflows, and to help manage the complexity of large-scale AI deployments.",
                "keypoints":["Research and Define Requirements: Conduct market research and gather requirements from potential users to determine what features and functionality the AI platform should have.",
                             "Develop Data Management System: Design and develop a data management system that can efficiently collect, store, and process large amounts of data.",
                             "Create Model Development Tools: Develop tools and libraries that make it easy for users to build, test, and iterate on machine learning models.",
                             "Deployment Infrastructure: Design and implement a deployment infrastructure that enables models to be deployed in a variety of environments, such as cloud, on-premises, or edge devices.",
                             "Monitor and Manage Models: Develop tools for monitoring and managing models in production, including performance and accuracy metrics, to ensure they continue to deliver value over time."],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","scikit-learn","Machine Learning","flask","Tabular Data","Visualization"],
                "Links":"",
                "category":"ML",
                "year":2018
        },
        {
                "title":"Synthetic image generation",
                "desc":"Synthetic image generation for medical analysis using Generative Adversarial Networks (GANs) is a process of creating artificial medical images using machine learning algorithms. GANs are a type of deep learning neural network that can generate images that are indistinguishable from real images. Synthetic images can be used to supplement real images and provide additional data for training machine learning models. The quality of the generated images has improved significantly in recent years, making it possible to generate high-quality images that can support accurate and reliable medical analysis.",
                "keypoints":[],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","PyTorch","Deep Learning","OpenCV","scikit-image","Pillow","Computer Vision","GAN"],
                "Links":"",
                "category":"DL",
                "year":2018
        },
        {
               "title":"Data Lineage toolbox",
                "desc":"A data lineage tool is a software solution that helps organizations track and manage the flow of data through their systems. The main purpose of a data lineage tool is to provide a clear understanding of how data is transformed and used over time, from its origin to its final destination. This helps organizations to meet regulatory requirements, improve data governance, and ensure data quality. A data lineage tool typically provides visual representations of data flows, allows users to trace data lineage from source to target, and tracks metadata, such as data definitions, business rules, and data transformations. It also provides auditing and reporting capabilities to support compliance and risk management initiatives.",
                "keypoints":['Data Flow Visualization: A visual representation of the flow of data within an organization, including the relationships between data sources, data transformations, and data destinations.', 'Data Traceability: The ability to trace data from its origin to its final destination, providing a clear understanding of how data is used and transformed over time.', 'Metadata Management: The ability to capture, store, and manage metadata, including data definitions, business rules, and data transformations.', 'Impact Analysis: The ability to analyze the impact of changes to data sources, data transformations, or data destinations, and determine how these changes may affect the flow of data.', 'Auditing and Reporting: The ability to capture, store, and report on changes made to data and metadata, to support compliance and risk management initiatives.', 'Integration with Other Tools: Integration with other data management and governance tools, such as data quality tools, data dictionaries, and metadata repositories, to provide a complete picture of the data landscape.', 'User-Friendly Interface: A user-friendly interface that makes it easy for users to understand the flow of data and trace data lineage, even for non-technical users.', 'Scalability: The ability to scale to meet the needs of large organizations with complex data landscapes.', 'Security: Measures to ensure the security of sensitive data, such as encryption, access controls, and data masking.', 'Reporting and Analytics: The ability to generate reports and perform analytics on data lineage information to support decision making and continuous improvement initiatives.'],
                "role":"Machine Learning Engineer",
                "org": "Virtusa",
                "tech": ["python","neo4j",'airflow',"AWS"],
                "Links":"",
                "category":"Others",
                "year":2018
                
        },
        
]
    
AWARDS=[{
                 "text":"DP-900: Azure Fundamentals",
                 "link":"pdf/Microsoft DP-900.pdf",
                 "year":2023,
                 "org": "Microsoft",
                 "url": "img/Microsoft.png",
                 "subtitle":"Satya Narayana Nadella",
                 "name":"Steller Award"
        },
         {
                 "text":"Machine Learning Engineering for Production (MLOps)- Specialization",
                 "link":"pdf/Machine Learning Engineering for Production (MLOps).pdf",
                 "year":2022,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
          {
                 "text":"Applied Data Science with Python-Specialization",
                 "link":"pdf/Applied Data Science with Python-Specialization.pdf",
                 "year":2018,
                 "org": "University of Michigan",
                 "url": "img/UOM.jpeg",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
          {
                 "text":"Deep Learning",
                 "link":"pdf/Deep Learning.pdf",
                 "year":2018,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
          {
                 "text":"Sequence Models- Deep Learning",
                 "link":"pdf/Sequence Models.pdf",
                 "year":2018,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
           {
                 "text":"Convolutional Neural Networks- Deep Learning",
                 "link":"pdf/Convolutional Neural Networks.pdf",
                 "year":2018,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
            {
                 "text":"Microsoft Azure Machine Learning for Data Scientists",
                 "link":"pdf/Microsoft Azure Machine Learning for Data Scientists.pdf",
                 "year":2022,
                 "org": "Coursera",
                 "url": "img/Coursera-Logo_.png",
                 "subtitle":"Mircrosoft",
                 "name":"Steller Award"
        },
            {
                 "text":"Create Machine Learning Models in Microsoft Azure",
                 "link":"pdf/Create Machine Learning Models in Microsoft Azure.pdf",
                 "year":2022,
                 "org": "Coursera",
                 "url": "img/Coursera-Logo_.png",
                 "subtitle":"Mircrosoft",
                 "name":"Steller Award"
        },
            {
                 "text":"Machine Learning Data Lifecycle in Production",
                 "link":"pdf/Machine Learning Data Lifecycle in Production.pdf",
                 "year":2022,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
        {
                 "text":"Applied Social Network Analysis in Python",
                 "link":"pdf/Applied Social Network Analysis in Python.pdf",
                 "year":2018,
                 "org": "University of Michigan",
                 "url": "img/UOM.jpeg",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
        {
                 "text":"Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization",
                 "link":"pdf/Improving Deep Neural Networks: Hyperparameter Tuning, Regularization and Optimization.pdf",
                 "year":2022,
                 "org": "DeepLearning.AI",
                 "url": "img/deeplearningai.png",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
        {
                 "text":"Applied Text Mining in Python",
                 "link":"pdf/Applied Text Mining in Python.pdf",
                 "year":2018,
                 "org": "University of Michigan",
                 "url": "img/UOM.jpeg",
                 "subtitle":"Coursera",
                 "name":"Steller Award"
        },
        
]

ROLES_AND_RESPOSIBILITY=[
        {
                "org": "KMPG",
                "subtitle":"Assurance and Consulting Services LLP",
                "position": "Consultant",
                "summary": "Data scientist with a strong background in AI and predictive maintenance. Proven track record of delivering innovative solutions to clients in the metal and mining sector and Investment Banking. Skilled in working with cross-functional teams and using data-driven insights to drive business growth.",
                "keypoints":["Metal and Mining :Worked with clients in the metal and mining sector using AI for predictive maintenance, resulting in cost savings of 25Cr. Conducted data analysis and built predictive models. Collaborated with cross-functional teams to deliver solutions.",
                             "Internal Audit: Developed a finding writing tool for auditors using NLP and machine learning, providing grammatical error correction, text scoring, text rephrasing and segregation of findings into issue impact, background and observation. Collaborated with stakeholders to meet business requirements."],
                "awards":[
                        {
                                "link":"pdf/Indresh Bhattacharya_Rising Star award_1952138 (1).pdf",
                                "url": "img/KMPG.jpeg",
                                "name":"Rising Star"
                        },
                        {
                                "link":"pdf/Indresh Bhattacharya_Kudos_2006968 (1).pdf",
                                "url": "img/KMPG.jpeg",
                                "name":"Kudos"
                        }
                        
                ]
        },
        {
                "org":"PRM Fincon",
                "subtitle":"Services Pvt. Ltd.",
                "position": "Data Scientist",
                "summary": "At PRM Fincom, I was a Key contributor to multiple projects aimed at improving financial processes at PRM Fincom. Utilized machine learning techniques for document processing and management. Proactively solved technical challenges to ensure project success and meet timelines. Deep understanding of financial sector and leveraged expertise to drive cost savings and improve operational efficiencies for clients. Collaborated with cross-functional teams to understand business requirements and deliver effective solutions.",
                "keypoints":['I leveraged my expertise in NLP and computer vision to develop a solution for Named Entity Recognition using OCR for Financial Forms.',
                             'I collaborated with cross-functional teams to ensure that all projects were delivered on time and to the highest quality standards.',
                             'I also provided technical support to clients to ensure the successful implementation of these solutions in their workflows.'],
                "awards":[]
        },
        {
                 "org": "Virtusa",
                 "subtitle":"IT Services Company ",
                 "position": "Assosiate ML Engineer | ML Engineer",
                 "summary":"As a team leader, consulted with clients globally to access Agent-Atlas interaction data and successfully coordinated a team of 4 engineers. Developed end-to-end solutions utilizing machine learning and statistical methods to reduce error rate for agents from 9% to 2% in quarterly reports.",
                 "keypoints":[ 'Contributed to the development of a Data Lineage Utility which streamlined data tracking and management processes.', 
                              'Built an AI platform that provided advanced analytics and decision-making capabilities to clients.',
                              'Developed a Data Quality Platform that ensured the accuracy and completeness of client data.', 
                              'Utilized Generative Adversarial Networks (GANs) to develop a Synthetic Image Generation tool that allowed for the creation of realistic images.',
                              'Designed and implemented a Google Ads Validation Framework that ensured the compliance of Google Ads with relevant regulations.'],
                 "awards":[
                         {
                                "link":"pdf/Steller_Award_Virtusa.jpeg",
                                "url": "img/Virtusa_logo.jpeg",
                                "name":"Steller Award"
                        },
                        {
                                "link":"",
                                "url": "img/Virtusa_logo.jpeg",
                                "name":"Dream Team"
                        },
                 ]
        }
        
        
        
]


SOCIALS={
        "instra":"https://www.instagram.com/indreshbhattacharyya/",
        "linkedin": "https://www.linkedin.com/in/indresh-bhattacharya/",
        "git":"https://github.com/neelindresh"
}




'''Pymlpipe is a cutting-edge open-source machine learning operations (MLOps) toolkit built in Python. It aims to provide an all-in-one solution for managing and streamlining machine learning projects, from model training and deployment to infrastructure and data monitoring.

                        With Pymlpipe, users can benefit from features such as model monitoring, data monitoring, infrastructure monitoring, and data drift detection. This toolkit also integrates with other popular open-source tools such as DVC, Concept Drift Detection, and eXplainable Artificial Intelligence (XAI), making it a comprehensive solution for MLOps.

                        The model monitoring feature helps users keep track of their models' performance, ensuring they are working as expected and allowing for quick detection and resolution of any issues. The data monitoring feature allows users to track the quality of their data and detect any data drift, helping to avoid issues that could impact the model's performance.

                        The pipeline feature provides a streamlined process for managing model training and deployment, reducing the time and effort required to get models up and running. The infrastructure monitoring feature helps users keep an eye on their resources and ensures their systems are performing optimally.

                        Pymlpipe is designed to be user-friendly and accessible, making it a great choice for organizations and individuals looking for a comprehensive MLOps solution. With its integration with cutting-edge technologies and open-source tools, it is well-positioned to help users stay ahead of the curve and stay competitive in today's rapidly-evolving machine learning landscape.'''


"""

KMPG=[
        ['Worked with clients in the metal and mining sector to implement AI for predictive maintenance.',
                             'Utilized data analytics and machine learning algorithms to develop predictive models for equipment failure.', 
                             'Conducted root cause analysis of equipment failures and recommended maintenance strategies.',
                             'Delivered business results by saving 25 crores in revenue for the clients.','Led the development of a finding writing tool for auditors.', 
                             'Utilized NLP and computer vision to build a grammatical error correction and text scoring system.',
                             'Developed a text rephrasing system to ensure the findings description was clear and concise.',
                             'Segregated the findings description into categories such as issue impact, background, and observation.', 
                             'Worked closely with auditors to ensure the tool met their requirements and provided meaningful insights.'
        ],


        ['Worked closely with clients in the metal and mining sector to implement AI solutions for predictive maintenance, resulting in cost savings of 25Cr for the client.',
                'Conducted data analysis and built predictive models to identify areas for improvement and optimization.', 
                'Collaborated with cross-functional teams, including data scientists, engineers, and business analysts, to deliver high-quality solutions.', 
                'Contributed to the development of a finding writing tool for auditors.',
                'Collaborated with stakeholders to understand requirements and develop a solution that met the needs of the business.',
                'Utilized NLP and machine learning techniques to build a solution that provided grammatical error correction, text scoring, text rephrasing, and segregation of findings description into issue impact, background, and observation.'
                ]
]
PRM=[
        '''Worked at PRM Fincom as a [Position], delivering innovative solutions for financial document management. Successfully completed several projects, including Financial Document Extraction, Automatic Document Tagging, Financial Document Classification, and Distributed Architecture Framework. Utilized advanced NLP and computer vision techniques to build a Named Entity Recognition system using OCR for financial forms.

Collaborated with cross-functional teams to understand the business requirements and deliver effective solutions. Utilized data analysis and machine learning techniques to build robust models for document processing and management. Proactively identified and solved complex technical challenges to ensure project success and meet project timelines.

Gained a deep understanding of the financial sector, its processes, and regulations, and leveraged this expertise to drive cost savings and improve operational efficiencies for clients.''',
]
"""