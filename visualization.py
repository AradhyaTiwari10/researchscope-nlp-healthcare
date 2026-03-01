from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    fig, ax = plt.subplots()
    ax.imshow(wordcloud)
    ax.axis("off")
    
    return fig