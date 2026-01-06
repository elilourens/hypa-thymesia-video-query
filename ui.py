import gradio as gr
import cv2
from pathlib import Path
from PIL import Image
from video_query import VideoQuerySystem


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

    def index_video(self, video_file, frame_interval=1.5, progress=gr.Progress()):
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

            self.vqs.index_video(video_path, frame_interval, video_id)
            self.indexed_videos[video_id] = video_path
            progress(1.0, desc="Done!")
            return f"Successfully indexed '{video_id}'!"
        except Exception as e:
            return f"Error: {str(e)}"

    def get_frame(self, video_path, timestamp):
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            cap.release()
            if ret:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return None
        except:
            return None

    def search_videos(self, query, n_results=10, selected_video=None):
        if self.vqs is None:
            return [], "Please initialize first"
        if not query or query.strip() == "":
            return [], "Please enter a search query"
        if not self.indexed_videos:
            return [], "No videos indexed yet"

        try:
            video_id_filter = None if selected_video == "All Videos" else selected_video
            results = self.vqs.search(query, n_results, video_id_filter)

            if not results:
                return [], f"No results found for '{query}'"

            gallery_data = []
            for result in results:
                metadata = result['metadata']
                frame = self.get_frame(metadata['video_path'], metadata['timestamp'])
                if frame is not None:
                    caption = f"Video: {metadata['video_id']}\nTime: {metadata['timestamp']:.2f}s\nSimilarity: {result['similarity']:.3f}"
                    gallery_data.append((Image.fromarray(frame), caption))

            return gallery_data, f"Found {len(gallery_data)} results for '{query}'"
        except Exception as e:
            return [], f"Error: {str(e)}"

    def get_video_list(self):
        return ["All Videos"] if not self.indexed_videos else ["All Videos"] + list(self.indexed_videos.keys())

    def clear_database(self):
        if self.vqs is None:
            return "No database to clear"
        try:
            self.vqs.client.delete_collection(name="video_frames")
            self.vqs.collection = self.vqs.client.get_or_create_collection(
                name="video_frames", metadata={"hnsw:space": "cosine"}
            )
            self.indexed_videos = {}
            return "Database cleared!"
        except Exception as e:
            return f"Error: {str(e)}"


def create_ui():
    ui = VideoQueryUI()

    with gr.Blocks(title="Video Query System", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Video Query System")
        gr.Markdown("Upload videos and search with natural language queries")

        with gr.Row():
            with gr.Column():
                gr.Markdown("## Upload & Index")
                video_input = gr.Video(label="Drag video here", sources=["upload"])
                frame_interval = gr.Slider(0.5, 5.0, 1.5, 0.5, label="Frame Interval (seconds)")
                index_button = gr.Button("Index Video", variant="primary")
                index_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Database")
                clear_button = gr.Button("Clear Database", variant="stop")
                clear_status = gr.Textbox(label="Clear Status", interactive=False)

            with gr.Column():
                gr.Markdown("## Search")
                search_query = gr.Textbox(label="Search Query", placeholder="e.g., red balloon, sunset...")
                with gr.Row():
                    video_filter = gr.Dropdown(["All Videos"], value="All Videos", label="Filter")
                    num_results = gr.Slider(1, 20, 10, 1, label="Max Results")
                search_button = gr.Button("Search", variant="primary")
                search_status = gr.Textbox(label="Status", interactive=False)

        gr.Markdown("## Results")
        results_gallery = gr.Gallery(label="Matching Frames", columns=5, rows=2, object_fit="contain")

        def index_and_update(video_file, interval):
            status = ui.index_video(video_file, interval)
            return status, gr.Dropdown(choices=ui.get_video_list())

        def clear_and_update():
            status = ui.clear_database()
            return status, gr.Dropdown(choices=ui.get_video_list()), [], ""

        index_button.click(index_and_update, [video_input, frame_interval], [index_status, video_filter])
        clear_button.click(clear_and_update, [], [clear_status, video_filter, results_gallery, search_status])
        search_button.click(ui.search_videos, [search_query, num_results, video_filter], [results_gallery, search_status])
        search_query.submit(ui.search_videos, [search_query, num_results, video_filter], [results_gallery, search_status])
        app.load(ui.initialize_system, outputs=index_status)

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(share=False, server_name="127.0.0.1", server_port=7860)
