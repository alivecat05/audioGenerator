import streamlit as st  # 导入 Streamlit，用于构建 Web 应用
import audio_Gen

import RAG
def main():
    flag = 0
    st.set_page_config(page_icon=":musical_note", page_title="Music Generator")
    # 网站大标题
    st.title("Text2Music音频生成器")
    
    # 添加可扩展的解释区域
    with st.expander("介绍"):
        st.write("Text2Music是一款基于audiocraft音频模型与浪潮源大模型的根据提示词生成音频的应用, 您可以在下方输入想要的声音, 稍等片刻就会有惊艳的声音给到你。")


    # 用户输入区域，用于输入描述文本
    text_area = st.text_area("请输入提示词：")
    
    time_slider = st.slider("选择生成时长（秒）", 2, 20, 20)
    if st.button("RAG ON")==1:
        theme = st.text_area("请输入风格：")==1
        RAG_prompt = RAG.prompt_enhance_RAG(text_area,theme)
        flag = 1


    # 时长滑块，用于选择音乐生成的时长
    # 如果用户提供了描述和时长
    if text_area and time_slider:
        # 显示用户输入的描述和时长
        st.json(
            {
                "描述": text_area,
                "时长": time_slider
            }
        )
        # 生成音乐子标题
        st.subheader("正在生成音乐...")
        if flag==1:
            newprompt = RAG_prompt
        else:
            newprompt = text_area
            
            
        # 生成音乐张量
        music_tensor = audio_Gen.generate_music_tensors(newprompt, time_slider)
        print("Music Tensors:", music_tensor)
        # 保存生成的音乐文件
        audio_filepath = audio_Gen.save_audio(music_tensor)
        # 在 Streamlit 中播放音频
        audio_file = open(audio_filepath, 'rb')
        audio_byte = audio_file.read()
        st.audio(audio_byte)
        # 显示下载链接，允许用户下载生成的音频文件
        st.markdown(audio_Gen.get_binary_file_html(audio_filepath, '音频文件'), unsafe_allow_html=True)

# 运行主函数
if __name__ == "__main__":
    main()