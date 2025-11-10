#!pip install subprocess ollama PySide6

import subprocess
import ollama
import sys
import socket
import os
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QLabel, QComboBox, QTextEdit, QPushButton, QMessageBox, QCheckBox,
    QSizePolicy, QProgressBar, QTabWidget, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, Signal, Slot, QThread, QObject
from PySide6.QtGui import QTextOption, QPixmap, QDragEnterEvent, QDropEvent, QClipboard


# Constants
RESPONSE_CONTENT_OLLAMA_FILE = "content_out@ollama.md"
TEMP_IMAGE_FILE = "temp.jpg"
OLLAMA_HOST = "localhost"  # Default Ollama host
OLLAMA_PORT = 11434  # Default Ollama port


def get_ollama_models():
    """Retrieves a list of available Ollama models using 'ollama list'.
    Handles potential errors gracefully and provides informative messages.
    """
    try:
        output = subprocess.check_output(['ollama', 'list'], text=True, stderr=subprocess.PIPE)
        model_names = [
            detail.split()[0]
            for detail in output.split('\n')
            if detail and detail.split()[0] != "NAME"  # Prevent empty strings and header from being processed
        ]
        return model_names
    except subprocess.CalledProcessError as e:
        print(f"Error listing models: {e}")
        print(f"Ollama output: {e.output}")
        return []
    except FileNotFoundError:
        print("Error: Ollama command not found. Ensure Ollama is installed and in your system's PATH.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []


def get_response(system_message, user_input, llm_model, images=None):
    """Retrieves a response from the specified Ollama model.
    Includes error handling for API requests.
    """
    messages = [{'role': 'system', 'content': system_message}] if system_message else []
    messages.append({'role': 'user', 'content': user_input})
    if images:
        messages[-1]['images'] = images

    try:
        response = ollama.chat(model=llm_model, messages=messages)
        return response['message']['content']
    except Exception as e:
        print(f"Error getting response from Ollama: {e}")
        return f"Error: Could not get response from {llm_model}. Check the console for details."


def save_response(response_content, filename=RESPONSE_CONTENT_OLLAMA_FILE):
    """Appends the chatbot's response to a file.
    Includes basic error handling.
    """
    try:
        with open(filename, "w") as file:
            file.write(response_content + "\n")
    except IOError as e:
        print(f"Error saving response to file: {e}")


def is_port_open(host, port):
    """Check if a port is open on the specified host.
    Improved error handling and reporting.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)  # Set a timeout to prevent indefinite blocking
        try:
            s.connect((host, port))
            return True
        except socket.timeout:
            print(f"Connection to {host}:{port} timed out.")
            return False
        except socket.error as e:
            print(f"Connection to {host}:{port} failed: {e}")
            return False
        except Exception as e:
            print(f"An unexpected error occurred while checking port: {e}")
            return False


class Worker(QObject):
    """Worker thread for running Ollama queries."""
    progress_changed = Signal(int)
    result_ready = Signal(str)
    error_occurred = Signal(str)  # Signal to report errors

    def __init__(self, system_message, user_input, selected_model, images=None):
        super().__init__()
        self.system_message = system_message
        self.user_input = user_input
        self.selected_model = selected_model
        self.images = images

    @Slot()
    def run(self):
        """Queries the Ollama model and emits the response or an error."""
        try:
            # Simulating progress updates
            for progress in range(0, 100, 10):
                self.progress_changed.emit(progress)
                QThread.sleep(1)  # Simulate task delay

            response = get_response(self.system_message, self.user_input, self.selected_model, self.images)
            self.progress_changed.emit(100)  # Ensure progress reaches full

            self.result_ready.emit(response)
        except Exception as e:
            print(f"Error in worker thread: {e}")
            self.error_occurred.emit(str(e))
            self.progress_changed.emit(0)  # Reset progress on error


class DragDropImageView(QGraphicsView):
    """A widget for drag-and-drop image handling."""

    image_dropped = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image_path = None  # Store the currently displayed image path

    def display_image(self, image_path):
        """Displays the given image in the graphics view."""
        try:
            self.scene.clear()
            pixmap = QPixmap(image_path)
            if pixmap.isNull():
                raise ValueError(f"Could not load image from {image_path}")

            # Scale the pixmap to fit within the view's size while maintaining aspect ratio
            item = QGraphicsPixmapItem(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.scene.addItem(item)
            self.setSceneRect(self.scene.itemsBoundingRect())  # Adjust scene size to fit image
            self.image_path = image_path  # Store path to the displayed image

        except Exception as e:
            print(f"Error displaying image: {e}")
            QMessageBox.critical(self, "Error", f"Could not display image: {e}")

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handles drag enter for image files."""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handles drop events for image files."""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):  # added more image types
                self.image_dropped.emit(file_path)
                break
        else:  # No acceptable file found
            QMessageBox.warning(self, "Warning", "Only image files (png, jpg, jpeg, gif, bmp) are supported.")

    def paste_image_from_clipboard(self):
        """Handles pasting an image from the system clipboard."""
        clipboard = QApplication.clipboard()
        mime_data = clipboard.mimeData()
        if mime_data and mime_data.hasImage():
            try:
                image = clipboard.image()
                image.save(TEMP_IMAGE_FILE, 'JPEG')
                self.image_dropped.emit(TEMP_IMAGE_FILE)
            except Exception as e:
                print(f"Error pasting image from clipboard: {e}")
                QMessageBox.critical(self, "Error", f"Could not paste image from clipboard: {e}")
        else:
            QMessageBox.warning(self, "Warning", "No image found in clipboard.")

    def get_image_path(self):
        """Returns the path of the currently displayed image or None if no image is displayed."""
        return self.image_path


class OllamaChatbotApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.check_ollama_available()

    def init_ui(self):
        """Initializes the user interface."""
        self.setWindowTitle("Ollama Vision")
        self.setGeometry(100, 100, 600, 400)  # Adjusted default window size for improved layout

        # Main layout
        self.main_layout = QVBoxLayout(self)

        # Tabs
        self.tabs = QTabWidget(self)
        self.main_layout.addWidget(self.tabs)

        # Add tabs
        self.add_prompt_tab()
        self.add_vision_tab()

        # Reload models now that both tabs have been added
        self.reload_models()

        self.setLayout(self.main_layout)

    def add_prompt_tab(self):
        """Adds the "Prompt" tab."""
        self.prompt_tab = QWidget()
        self.tabs.addTab(self.prompt_tab, "Prompt")
        layout = QVBoxLayout(self.prompt_tab)

        layout.addWidget(QLabel("Available Ollama models:", self))

        # Model selection
        self.model_combo = QComboBox(self)
        layout.addWidget(self.model_combo)

        # Checkbox for system prompt visibility
        self.show_checkbox = QCheckBox("Show System Prompt", self)
        self.show_checkbox.stateChanged.connect(self.toggle_system_input)
        layout.addWidget(self.show_checkbox)

        # System input
        self.system_input = QTextEdit(self)
        self.system_input.setPlaceholderText("System Prompt")
        self.system_input.setVisible(False)
        layout.addWidget(self.system_input)

        # Question input
        layout.addWidget(QLabel("Enter Question(s)", self))
        self.question_input = QTextEdit(self)
        self.question_input.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.question_input)

        # Submit button
        self.submit_button = QPushButton("Submit", self)
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

        # Results display
        self.result_label = QTextEdit(self)
        self.result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.result_label.setWordWrapMode(QTextOption.WordWrap)
        self.result_label.setReadOnly(True)
        layout.addWidget(self.result_label)

        # Reset button
        self.reset_button = QPushButton("Reset", self)
        self.reset_button.clicked.connect(self.reset_ui)
        layout.addWidget(self.reset_button)

    def add_vision_tab(self):
        """Adds the "Vision" tab."""
        self.vision_tab = QWidget()
        self.tabs.addTab(self.vision_tab, "Vision")
        layout = QVBoxLayout(self.vision_tab)

        layout.addWidget(QLabel("Available Ollama models:", self))

        # Model selection
        self.vision_model_combo = QComboBox(self)
        layout.addWidget(self.vision_model_combo)

        # Image preview area
        self.image_view = DragDropImageView(self)
        self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)  # Make the view resizable
        self.image_view.setMinimumSize(150, 150)  # Optionally set a minimum size
        self.image_view.image_dropped.connect(self.handle_image)
        layout.addWidget(self.image_view)

        # Paste from clipboard button
        self.paste_button = QPushButton("Paste Image from Clipboard", self)
        self.paste_button.clicked.connect(self.image_view.paste_image_from_clipboard)
        layout.addWidget(self.paste_button)

        # Process button
        self.process_image_button = QPushButton("Process Image", self)
        self.process_image_button.clicked.connect(self.process_image)
        layout.addWidget(self.process_image_button)

        # Vision progress bar
        self.vision_progress_bar = QProgressBar(self)
        self.vision_progress_bar.setRange(0, 100)
        layout.addWidget(self.vision_progress_bar)

        # Results display
        self.vision_result_label = QTextEdit(self)
        self.vision_result_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.vision_result_label.setWordWrapMode(QTextOption.WordWrap)
        self.vision_result_label.setReadOnly(True)
        layout.addWidget(self.vision_result_label)

        # Clear Image button
        self.clear_image_button = QPushButton("Clear Image", self)
        self.clear_image_button.clicked.connect(self.clear_vision_image)
        layout.addWidget(self.clear_image_button)

    def reload_models(self):
        """Reloads the model list in the combo boxes."""
        model_list = get_ollama_models()
        model_list.sort()
        self.model_combo.clear()
        self.model_combo.addItems(model_list if model_list else ["No models available"])
        self.vision_model_combo.clear()
        self.vision_model_combo.addItems(model_list if model_list else ["No models available"])

    def toggle_system_input(self):
        """Toggles the visibility of the system input."""
        self.system_input.setVisible(self.show_checkbox.isChecked())

    @Slot()
    def on_submit(self):
        """Handles the 'Submit' button click in the 'Prompt' tab."""
        selected_model = self.model_combo.currentText()
        system_message = self.system_input.toPlainText()
        user_input = self.question_input.toPlainText()

        if not selected_model or not user_input:
            QMessageBox.warning(self, "Warning", "Please select a model and enter a question.")
            return

        self.submit_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.result_label.clear()  # Clear previous results

        # Run worker thread for LLM query
        self.worker_thread = QThread()
        self.worker = Worker(system_message, user_input, selected_model)
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress_changed.connect(self.progress_bar.setValue)
        self.worker.result_ready.connect(self.display_prompt_result)
        self.worker.error_occurred.connect(self.handle_worker_error)  # Handle errors from the worker thread
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # Clean up thread when finished
        self.worker_thread.start()

    @Slot(str)
    def handle_image(self, image_path):
        """Handles an image being dropped or pasted."""
        self.image_view.display_image(image_path)

    @Slot()
    def process_image(self):
        """Processes the image using Ollama Vision."""
        image_path = self.image_view.get_image_path()  # Get the path to the displayed image
        if not image_path:
            QMessageBox.warning(self, "Warning", "Please drop or paste an image first.")
            return

        selected_model = self.vision_model_combo.currentText()
        if not selected_model:
            QMessageBox.warning(self, "Warning", "Please select a model.")
            return

        self.vision_result_label.clear()  # Clear previous results
        self.vision_progress_bar.setValue(0)  # Reset vision progress bar

        # Run worker thread for vision query
        self.worker_thread = QThread()
        self.worker = Worker("", "Extract text from this image:", selected_model, [image_path])
        self.worker.moveToThread(self.worker_thread)
        self.worker.progress_changed.connect(self.vision_progress_bar.setValue)
        self.worker.result_ready.connect(self.display_vision_result)
        self.worker.error_occurred.connect(self.handle_worker_error)
        self.worker_thread.started.connect(self.worker.run)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)  # Clean up thread when finished
        self.worker_thread.start()

    @Slot(str)
    def display_prompt_result(self, response):
        """Displays the result in the 'Prompt' tab."""
        self.result_label.setPlainText(response)
        save_response(response)
        self.submit_button.setEnabled(True)
        self.worker_thread.quit()
        self.worker_thread.wait()

    @Slot(str)
    def display_vision_result(self, response):
        """Displays the result in the 'Vision' tab."""
        self.vision_result_label.setPlainText(response)
        save_response(response)
        self.worker_thread.quit()
        self.worker_thread.wait()

    @Slot()
    def reset_ui(self):
        """Resets the UI for the 'Prompt' tab."""
        self.system_input.clear()
        self.question_input.clear()
        self.result_label.clear()
        self.progress_bar.setValue(0)
        self.submit_button.setEnabled(True)

    @Slot()
    def clear_vision_image(self):
        """Clears the image from the vision tab."""
        self.image_view.scene.clear()
        self.image_view.image_path = None  # Clear the stored image path
        self.vision_result_label.clear()  # Clear results as well

    @Slot(str)
    def handle_worker_error(self, error_message):
        """Handles errors that occur in the worker thread."""
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")
        self.submit_button.setEnabled(True)  # Re-enable button on error
        self.process_image_button.setEnabled(True)  # Re-enable the vision button
        if self.tabs.currentIndex() == 0:
            self.progress_bar.setValue(0)  # Reset progress bar if on the "Prompt" tab
        self.worker_thread.quit()
        self.worker_thread.wait()

    def check_ollama_available(self):
        """Checks if Ollama is running and displays a warning if not."""
        if not is_port_open(OLLAMA_HOST, OLLAMA_PORT):
            QMessageBox.warning(
                self,
                "Ollama Not Available",
                "Ollama does not seem to be running. Please ensure Ollama is running and accessible "
                f"on {OLLAMA_HOST}:{OLLAMA_PORT}.",
            )


def main():
    """Main entry point of the application."""
    app = QApplication(sys.argv)
    window = OllamaChatbotApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
