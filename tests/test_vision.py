from src.vision.image_analyzer import analyze_image


image_path = (
    "data/extracted_images/"
    "page_9_image_4.jpeg"
)


description = analyze_image(
    image_path
)


print("\n")
print("=" * 60)
print("VISION ANALYSIS")
print("=" * 60)
print("\n")

print(description)

print("\n")
print("=" * 60)