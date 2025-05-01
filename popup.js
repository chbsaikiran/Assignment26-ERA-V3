document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('textInput');
    const imageInput = document.getElementById('imageInput');
    const preview = document.getElementById('preview');
    const previewContainer = document.getElementById('previewContainer');
    const sendBtn = document.getElementById('sendBtn');

    // Handle image preview
    imageInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                preview.src = e.target.result;
                previewContainer.style.display = 'block';
                textInput.value = ''; // Clear text input when image is selected
            }
            reader.readAsDataURL(file);
        }
    });

    // Handle text input
    textInput.addEventListener('input', function() {
        if (textInput.value) {
            preview.src = '';
            previewContainer.style.display = 'none';
            imageInput.value = ''; // Clear image input when text is entered
        }
    });

    // Handle send button click
    sendBtn.addEventListener('click', async function() {
        const text = textInput.value;
        const imageFile = imageInput.files[0];

        if (!text && !imageFile) {
            alert('Please enter text or select an image');
            return;
        }

        try {
            if (text) {
                // Send text
                const response = await fetch('http://localhost:8000/process_text', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                if (!response.ok) throw new Error('Failed to send text');
                alert('Text sent successfully!');
            } else if (imageFile) {
                // Send image
                const formData = new FormData();
                formData.append('file', imageFile);

                const response = await fetch('http://localhost:8000/process_image', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Failed to send image');
                alert('Image sent successfully!');
            }

            // Clear inputs after successful send
            textInput.value = '';
            imageInput.value = '';
            preview.src = '';
            previewContainer.style.display = 'none';

        } catch (error) {
            alert('Error: ' + error.message);
        }
    });
}); 