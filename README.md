
# 🎬 Video Analysis Agent

AI-powered Video Analysis Agent with LLM integration and RAG capabilities for comprehensive video content analysis.

## Features

- **Frame-by-Frame Analysis**: Extracts and analyzes video frames using OpenAI's Vision API
- **Object Detection**: Identifies objects, people, and scenes in video content
- **Emotion Recognition**: Detects emotions and expressions from video frames
- **Text Extraction**: OCR capabilities to extract text from video frames
- **RAG Integration**: Build knowledge base from video analysis for Q&A
- **Sentiment Analysis**: Overall sentiment analysis of video content
- **Interactive UI**: Streamlit-based web interface for easy interaction

## Requirements

- Python 3.11+
- OpenAI API Key
- Video files (mp4, avi, mov, mkv)

## Installation

1. Clone this repository:
```bash
git clone <your-repo-url>
cd video-analysis-agent
```

2. Install dependencies:
```bash
pip install -e .
```

3. Set up your OpenAI API key:
   - Get an API key from [OpenAI Platform](https://platform.openai.com/api-keys)
   - Enter it in the Streamlit interface when prompted

## Usage

1. Run the Streamlit application:
```bash
streamlit run main.py --server.address=0.0.0.0 --server.port=5000
```

2. Open your browser and navigate to the application URL

3. Enter your OpenAI API key in the sidebar

4. Upload a video file (supported formats: mp4, avi, mov, mkv)

5. Click "Analyze Video" to start the analysis

6. Explore the results in different tabs:
   - **Summary**: Overall video summary and key topics
   - **Frame Analysis**: Detailed frame-by-frame breakdown
   - **Insights**: Key insights extracted from the video
   - **Q&A**: Ask questions about the video content
   - **Metrics**: Analysis statistics and metrics

## Architecture

The Video Analysis Agent consists of several key components:

- **VideoAnalysisAgent**: Main class handling video processing and analysis
- **Frame Extraction**: Extracts frames at regular intervals from video
- **Vision Analysis**: Uses OpenAI GPT-4 Vision for frame analysis
- **RAG System**: Builds searchable knowledge base from analysis results
- **Streamlit UI**: User-friendly web interface

## API Usage

The agent can also be used programmatically:

```python
from main import VideoAnalysisAgent

# Initialize agent
agent = VideoAnalysisAgent(api_key="your-openai-api-key")

# Analyze video
result = await agent.analyze_video("path/to/video.mp4", "video_id")

# Query video content
answer = agent.query_video_content("What objects are visible in the video?")
```

## Configuration

Key configuration options:

- **Frame Interval**: Adjust frame extraction interval (default: 30 seconds)
- **Analysis Depth**: Control detail level of frame analysis
- **RAG Parameters**: Customize knowledge base chunking and retrieval

## Limitations

- Analysis time depends on video length and OpenAI API response times
- Large videos may require significant processing time
- Requires stable internet connection for OpenAI API calls
- API usage costs depend on video length and analysis depth

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open a GitHub issue or contact the development team.
