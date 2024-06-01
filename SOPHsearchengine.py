from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

class SOPSearchEngine:
    _dictionary = [
        "Machine learning is a discipline within artificial intelligence.",
        "Artificial Intelligence encompasses various fields.",
        "Natural Language Processing resides within the AI umbrella.",
        "Machine learning enables systems to glean insights from data.",
        "Python remains a preferred programming language for machine learning.",
        "Deep learning has revolutionized image and speech recognition.",
        "Reinforcement learning is gaining traction in decision-making systems.",
        "Data analytics is crucial for modern businesses.",
        "Statistical methods provide the backbone for many machine learning algorithms.",
        "Transfer learning allows pre-trained models to adapt to new tasks.",
        "Oishi Sen LinkedIn profile.",
        "Sumana Kundu Facebook profile.",
        "Asir's Instagram profile.",
        "Shivam's Instagram profile.",
        "Pritam's Instagram profile.",
        "Check out Netflix.",
        "Visit YouTube.",
        "Listen on Spotify."
    ]

    _websites = [
        "https://www.geeksforgeeks.org/machine-learning/",
        "https://www.tutorialspoint.com/artificial_intelligence/index.htm",
        "https://www.nltk.org/",
        "https://www.datasciencecentral.com/",
        "https://www.python.org/",
        "https://www.tensorflow.org/overview",
        "https://openai.com/research/",
        "https://towardsdatascience.com/",
        "https://statisticsbyjim.com/basics/",
        "https://huggingface.co/",
        "https://www.linkedin.com/in/oishi-sen-29538222a/?originalSubdomain=in",
        "https://www.facebook.com/sumana.kundu.3?mibextid=ZbWKwL",
        "https://instagram.com/asir.xo?igshid=MzRlODBiNWFlZA==",
        "https://instagram.com/shivamx__?igshid=MzRlODBiNWFlZA==",
        "https://instagram.com/__pritam_._?igshid=MzRlODBiNWFlZA==",
        "https://www.netflix.com/",
        "https://www.youtube.com/",
        "https://www.spotify.com/"
    ]

    _image_urls = [
        "https://www.geeksforgeeks.org/wp-content/uploads/machineLearning3.png",
        "https://www.tutorialspoint.com/artificial_intelligence/images/artificial_intelligence.jpg",
        "https://www.nltk.org/_static/nltk_logo.png",
        "https://www.datasciencecentral.com/profiles/blogs/logo2.jpg",
        "https://www.python.org/static/community_logos/python-logo.png",
        "https://www.tensorflow.org/images/tf_logo_social.png",
        "https://openai.com/v5/assets/images/brand/openai/logo/social.png",
        "https://towardsdatascience.com/icons/icon-48x48.png",
        "https://statisticsbyjim.com/wp-content/uploads/2017/03/cropped-StatisticsByJimLogo.png",
        "https://huggingface.co/front/assets/huggingface_logo.svg",
        "https://static-exp1.licdn.com/sc/h/al2o9zrvru7aqj8e1x2rzsrca",  # LinkedIn Logo
        "https://static.xx.fbcdn.net/rsrc.php/y8/r/dF5SId3UHWd.svg",      # Facebook Logo
        "https://instagram.com/static/images/ico/favicon.ico/36b3ee2d910d.png",
        "https://instagram.com/static/images/ico/favicon.ico/36b3ee2d910d.png",
        "https://instagram.com/static/images/ico/favicon.ico/36b3ee2d910d.png",
        "https://assets.stickpng.com/images/580b57fcd9996e24bc43c529.png",  # Netflix Logo
        "https://assets.stickpng.com/images/580b57fcd9996e24bc43c545.png",  # YouTube Logo
        "https://cdn.worldvectorlogo.com/logos/spotify-1.svg"               # Spotify Logo
    ]

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.doc_vectors = self.vectorizer.fit_transform(self._documents)

    def search(self, query):
        query_vector = self.vectorizer.transform([query])
        cosine_similarities = linear_kernel(query_vector, self.doc_vectors).flatten()
        related_docs_indices = cosine_similarities.argsort()[:-5:-1]  # Get top 5 matches
        results = [{"doc": self._documents[i], "website": self._websites[i], "image_url": self._image_urls[i]}
                   for i in related_docs_indices if cosine_similarities[i] > 0]

        if not results:
            return [{"doc": "No matches found", "website": "", "image_url": ""}]
        
        return results