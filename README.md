<h1>Abstract or Overview:</h1>
The goal of this project is to construct a web application that summarises CNN articles using extractive summarization techniques. The application allows users to enter a CNN article URL and receive a brief summary of the story's key elements. 
This tool can help busy professionals, students, and anybody else quickly comprehend the content of a news piece. It allows users to quickly extract vital information from lengthy articles, saving time and effort.


<h1>Data Description:</h1>
This project's dataset comprises of CNN stories contained in a CSV file named 'WASHINGTON (CNN)1.csv'. Each row in the dataset represents an article, with the main text saved in the 'article' column, highlights in the highlights column and unique Id in the Id column. Before analysis, the text was preprocessed with tokenization, stop word removal, and TF-IDF vectorization.


<h1>Algorithm Description:</h1>
The web software uses extractive summarising techniques to create summaries of CNN articles. It computes TF-IDF scores for each sentence in the article and uses the top-ranking sentences to create the summary. Additionally, it uses the BART model to fine-tune the summarization process.


<h1>Tools Used:</h1>
Streamlit: Used to create the web application interface.
Requests: Utilized for fetching article content from URLs.
BeautifulSoup: Employed for parsing HTML content from web pages.
Transformers: Used for fine-tuning the BART model for summarization.
Pandas: Utilized for data manipulation and analysis.
NLTK: Used for natural language processing tasks such as tokenization and stop word removal.
Scikit-learn: Employed for TF-IDF vectorization.

<h1>Ethical Concerns:</h1>
One ethical concern with this use is the possibility of prejudice throughout the summary process. The algorithm may prioritise some information over others, resulting in a biased summary. To reduce this risk, the summarization model must be evaluated and fine-tuned on a regular basis to ensure its fairness and correctness. Users should also be aware that the app's summaries are generated algorithmically and may not capture all of the nuances included in the original articles.
Link to the website: https://mvp-strea-eaxvuge9zufej5xbqjeqzy.streamlit.app/
