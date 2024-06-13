document.addEventListener('DOMContentLoaded', function() {
    // Get the form and result div
    const form = document.getElementById('prediction-form');
    const resultDiv = document.getElementById('prediction-result');

    // Add an event listener for form submission
    form.addEventListener('submit', async (event) => {
        event.preventDefault(); // Prevent the form from submitting normally
        
        // Gather form data
        const formData = new FormData(form);
        const data = {};
        formData.forEach((value, key) => data[key] = value);
        
        try {
            // Send form data to Flask
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Accept': 'application/json',
                },
                body: new URLSearchParams(data) // Convert form data to URL-encoded string
            });
            
            // Parse the response
            const result = await response.json();
            
            // Update the result div with the prediction
            resultDiv.innerText = `Prediction: ${result.prediction}`;
        } catch (error) {
            resultDiv.innerText = 'Error: Could not get prediction. Please try again.';
        }
    });
});
