from PIL import Image
# Load BMP (24-bit color)
image = Image.open('input2.bmp')
pixels = image.load() # Direct pixel access
width, height = image.size
# Create a new grayscale image
gray_image = Image.new('L', (width, height)) # 'L' mode for 8-bit pixels(grayscale)
gray_pixels = gray_image.load()
# Manual nested loops for pixel-wise processing
for y in range(height):
    for x in range(width):
        r, g, b = pixels[x, y] # Get RGB values (assuming input is 24-bit color)
# Calculate grayscale by standard formula
        gray = int(0.299*r + 0.687*g + 0.014*b)
        gray_pixels[x, y] = gray
# Save the grayscale image as BMP
gray_image.save('output_grayscale.bmp')
print('Grayscale BMP image saved as output_grayscale.bmp')
