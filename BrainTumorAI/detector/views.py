import os

from django.conf import settings
from django.shortcuts import render

from .predict import predict_image


def home(request):

    if request.method == "POST":

        uploaded_image = request.FILES.get("image")

        if not uploaded_image:
            return render(
                request,
                "home.html",
                {
                    "error": "Please select an MRI image."
                }
            )

        # Create media/temp folder
        upload_dir = os.path.join(
            settings.MEDIA_ROOT,
            "temp"
        )

        os.makedirs(
            upload_dir,
            exist_ok=True
        )

        # Save uploaded image
        image_path = os.path.join(
            upload_dir,
            uploaded_image.name
        )

        with open(image_path, "wb+") as destination:

            for chunk in uploaded_image.chunks():
                destination.write(chunk)

        print("FILE RECEIVED:", uploaded_image.name)

        # AI prediction
        prediction, confidence = predict_image(
            image_path
        )

        print("PREDICTION:", prediction)
        print("CONFIDENCE:", confidence)

        return render(
            request,
            "result.html",
            {
                "prediction": prediction,
                "confidence": round(confidence, 2),
                "image_name": uploaded_image.name,
            }
        )

    return render(
        request,
        "home.html"
    )