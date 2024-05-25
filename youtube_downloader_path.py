from pytube import YouTube
import os
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
with st.sidebar:
    st.title('YouTube Downloader')

    add_selectbox = st.sidebar.selectbox(
    "Select what you want to download?",
    ("Video", "audio"))

    st.markdown ('''
    ## About 
    You can now download your favorite media from YouTube by just providing the YouTube link to it;
    and selecting to download the video 
    or audio file of the media provided:

    - [streamlit](https://streamlit.io)             
    - [pytube](https://pytube.io/)             

    ''')

    add_vertical_space(5)
    st.write('Made by [Dev Tobs](https://twitter.com/AnyigorTobias)')

def Download():
    #add radio buttons and modify codes to allow user select the type of file needed
    # this will add a header to the streamlit app
    st.header("Youtube downloader")
    if add_selectbox == 'Video':

        # this brings up an input box for the url
        youtube_url = st.text_input("Enter the YouTube URL")

        # Input path
        save_dir = st.text_input("Select the directory to save the file:", value=os.path.expanduser("~"))


        # the st.radio brings up selection buttons for selectiong video resolution
        genre = st.radio(
                    "Select the resolution you will like to download",
                    ["Highest Resolution", "720p", "480p", "360p", "240p", "144p"]
                    )
        # this brings up a download button you can click on to initiate video download
        if st.button("Download video"):
                try:
                    youtubeObject = YouTube(youtube_url)


                    if genre == "Highest Resolution":
                        youtubeObject = youtubeObject.streams.get_highest_resolution()
                    elif genre == "720p":
                        youtubeObject = youtubeObject.streams.get_by_resolution("720p")
                    elif genre == "480p":
                        youtubeObject = youtubeObject.streams.get_by_resolution("480p")
                    elif genre == "360p":
                        youtubeObject = youtubeObject.streams.get_by_resolution("360p")

                    else:
                        youtubeObject = youtubeObject.streams.get_by_resolution("144p")
                     # creates a directory for downloads   
                    if youtubeObject:  
                        #save_dir = st.text_input("Select the directory to save the file:", value=os.path.expanduser("~"))
                        #'output/.mp4'
                        #os.makedirs('output', exist_ok=True)
                        youtubeObject.download(output_path=save_dir)
                        st.success("Download completed successfully.")
                    else:
                        st.error("No suitable video stream found for the selected resolution, try another resolution")

                except Exception as e:
                    st.error(f"An error occurred: {e}")


if add_selectbox == 'audio':
        youtube_url = st.text_input("Enter the YouTube URL")
        save_dir = st.text_input("Select the directory to save the file:", value=os.path.expanduser("~"))

        if st.button("Download audio"):

            try:
                file = ""

                youtubeObject = YouTube(youtube_url)
                audio = youtubeObject.streams.filter(only_audio = True).first()

                title = os.path.basename(file)[:-4]

                #save_dir = 'output/.mp3'
                #os.makedirs('output', exist_ok = True)
                file = audio.downloutput_path = save_dir
                st.success("Download completed successfully")
            except Exception as e:
                st.error(f"An error occurred: {e}")



if __name__ == '__main__':
    Download()