import gradio as gr
from pathlib import Path
from PIL import Image
from src.models.video_query import VideoQuerySystem
from src.video.processor import VideoProcessor


class VideoQueryUI:
    def __init__(self):
        self.vqs = None
        self.indexed_videos = {}

    def initialize_system(self):
        if self.vqs is None:
            try:
                self.vqs = VideoQuerySystem()
                return "System initialized!"
            except Exception as e:
                return f"Error: {str(e)}"
        return "Already initialized"

    def index_video(self, video_file, frame_interval=1.5, skip_solid_frames=True, detect_scenes=True, transcribe=True, progress=gr.Progress()):
        if self.vqs is None:
            self.initialize_system()
        if video_file is None:
            return "Please upload a video file"

        try:
            progress(0, desc="Indexing...")
            video_path = video_file
            video_id = Path(video_path).stem

            if video_id in self.indexed_videos:
                return f"'{video_id}' already indexed!"

            self.vqs.index_video(video_path, frame_interval, video_id, skip_solid_frames, save_frames_to_disk=True, detect_scenes=detect_scenes, transcribe=transcribe)
            self.indexed_videos[video_id] = video_path
            progress(1.0, desc="Done!")
            return f"Successfully indexed '{video_id}'!"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_frame(self, video_path, timestamp):
        return VideoProcessor.get_frame_at_timestamp(video_path, timestamp)

    def search_videos(self, query, n_results=10, selected_video=None, diversify=True, diversity_weight=0.5, search_type="frames"):
        if self.vqs is None:
            return [], [], "Please initialize first"
        if not query or query.strip() == "":
            return [], [], "Please enter a search query"
        if not self.indexed_videos:
            return [], [], "No videos indexed yet"

        try:
            video_id_filter = None if selected_video == "All Videos" else selected_video

            frame_gallery = []
            transcript_results = []

            if search_type in ["frames", "both"]:
                frame_results = self.vqs.search(query, n_results, video_id_filter, diversify, diversity_weight)
                for result in frame_results:
                    metadata = result['metadata']
                    frame = self.get_frame(metadata['video_path'], metadata['timestamp'])
                    if frame is not None:
                        caption = f"Video: {metadata['video_id']}\nTime: {metadata['timestamp']:.2f}s\nSimilarity: {result['similarity']:.3f}"
                        if 'scene_id' in metadata:
                            caption += f"\nScene: {metadata['scene_id']}"
                        frame_gallery.append((Image.fromarray(frame), caption))

            if search_type in ["transcripts", "both"]:
                transcript_results_raw = self.vqs.search_transcripts(query, n_results, video_id_filter, diversify, diversity_weight)
                for result in transcript_results_raw:
                    metadata = result['metadata']
                    result_text = f"**Video:** {metadata['video_id']}\n**Time:** {metadata['start_time']:.2f}s - {metadata['end_time']:.2f}s\n**Similarity:** {result['similarity']:.3f}\n**Text:** {metadata['text']}\n\n---\n"
                    transcript_results.append(result_text)

            status = f"Found {len(frame_gallery)} frames and {len(transcript_results)} transcript matches for '{query}'"
            return frame_gallery, "\n".join(transcript_results), status
        except Exception as e:
            return [], [], f"Error: {str(e)}"

    def get_video_list(self):
        return ["All Videos"] if not self.indexed_videos else ["All Videos"] + list(self.indexed_videos.keys())

    def clear_database(self):
        if self.vqs is None:
            return "No database to clear"
        try:
            self.vqs.clear_all()
            self.indexed_videos = {}
            return "Database cleared!"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_stats(self):
        if self.vqs is None:
            return "0 frames | 0 transcripts"
        try:
            frame_count = self.vqs.get_frame_count()
            transcript_count = self.vqs.get_transcript_count()
            return f"{frame_count} frames | {transcript_count} transcript chunks"
        except Exception as e:
            return f"Error: {str(e)}"


def create_ui():
    ui = VideoQueryUI()

    with gr.Blocks(title="Video Query System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Video Query System")
        gr.Markdown("Upload videos and search with natural language queries")

        with gr.Tabs():
            # Upload Tab
            with gr.Tab("Upload & Index"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Video Upload")
                        video_input = gr.Video(label="Drag video here", sources=["upload"])

                        gr.Markdown("### Frame Extraction Settings")
                        frame_interval = gr.Slider(0.5, 5.0, 1.5, 0.5, label="Frame Interval (seconds)")
                        skip_solid = gr.Checkbox(value=True, label="Skip solid colors & gradients")
                        detect_scenes = gr.Checkbox(value=True, label="Detect scene changes")

                        gr.Markdown("### Transcription Settings")
                        transcribe_checkbox = gr.Checkbox(value=True, label="Transcribe audio (Whisper)")

                        index_button = gr.Button("Index Video", variant="primary", size="lg")
                        index_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column():
                        gr.Markdown("## Database Management")
                        stats_display = gr.Textbox(
                            label="Database Statistics",
                            interactive=False,
                            value="0 frames | 0 transcripts"
                        )

                        gr.Markdown("### Indexed Videos")
                        indexed_videos_list = gr.Dropdown(
                            ["All Videos"],
                            value="All Videos",
                            label="Videos in Database",
                            interactive=False
                        )

                        clear_button = gr.Button("Clear Database", variant="stop")
                        clear_status = gr.Textbox(label="Clear Status", interactive=False)

            # Search Tab
            with gr.Tab("Search"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## Search Settings")
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., red balloon, sunset, person speaking...",
                            lines=2
                        )

                        search_type = gr.Radio(
                            choices=["frames", "transcripts", "both"],
                            value="both",
                            label="Search Type",
                            info="Search visual frames, transcripts, or both"
                        )

                        video_filter = gr.Dropdown(
                            ["All Videos"],
                            value="All Videos",
                            label="Filter by Video"
                        )

                        num_results = gr.Slider(1, 20, 10, 1, label="Max Results per Type")

                        gr.Markdown("### Diversity Settings")
                        diversify_checkbox = gr.Checkbox(value=True, label="Enable Temporal Diversity")
                        diversity_slider = gr.Slider(
                            0, 1, 0.5, 0.1,
                            label="Diversity Weight",
                            info="0=relevance only, 1=diversity only"
                        )

                        search_button = gr.Button("Search", variant="primary", size="lg")
                        search_status = gr.Textbox(label="Status", interactive=False)

                    with gr.Column(scale=2):
                        gr.Markdown("## Results")

                        gr.Markdown("### Frame Results")
                        results_gallery = gr.Gallery(
                            label="Matching Frames",
                            columns=4,
                            rows=2,
                            object_fit="contain",
                            height="auto"
                        )

                        gr.Markdown("### Transcript Results")
                        transcript_results = gr.Markdown(
                            label="Matching Transcripts",
                            value="No results yet"
                        )

        # Event handlers
        def index_and_update(video_file, interval, skip_solid_frames, detect_scenes, transcribe):
            status = ui.index_video(video_file, interval, skip_solid_frames, detect_scenes, transcribe)
            stats = ui.get_stats()
            return status, gr.Dropdown(choices=ui.get_video_list()), stats

        def clear_and_update():
            status = ui.clear_database()
            stats = ui.get_stats()
            return status, gr.Dropdown(choices=ui.get_video_list()), [], "", "", stats

        def initialize_and_stats():
            status = ui.initialize_system()
            stats = ui.get_stats()
            return status, stats

        # Wire up events
        index_button.click(
            index_and_update,
            [video_input, frame_interval, skip_solid, detect_scenes, transcribe_checkbox],
            [index_status, indexed_videos_list, stats_display]
        )

        clear_button.click(
            clear_and_update,
            [],
            [clear_status, indexed_videos_list, results_gallery, transcript_results, search_status, stats_display]
        )

        search_button.click(
            ui.search_videos,
            [search_query, num_results, video_filter, diversify_checkbox, diversity_slider, search_type],
            [results_gallery, transcript_results, search_status]
        )

        search_query.submit(
            ui.search_videos,
            [search_query, num_results, video_filter, diversify_checkbox, diversity_slider, search_type],
            [results_gallery, transcript_results, search_status]
        )

        app.load(initialize_and_stats, outputs=[index_status, stats_display])

    return app
