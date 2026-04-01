import cv2

# List of template filenames
template_files = [
    r"Week9\template1.png",
    r"Week9\template2.png",
    r"Week9\template3.png",
    r"Week9\template4.png",
    r"Week9\template5.png"
]

# Load and compute SIFT keypoints/descriptors for each template
sift = cv2.SIFT_create()
templates = []
for filename in template_files:
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Cannot load {filename}")
        exit()
    kp, des = sift.detectAndCompute(img, None)
    templates.append({"name": filename, "image": img, "kp": kp, "des": des})

# Load the test image
test_img = cv2.imread(r"Week9\template.png", cv2.IMREAD_GRAYSCALE)
if test_img is None:
    print("Cannot load template")
    exit()

kp_test, des_test = sift.detectAndCompute(test_img, None)
if des_test is None:
    print("No descriptors in test image!")
    exit()

# Initialize matcher
bf = cv2.BFMatcher()

# For each template, compute the number of good matches with the test image
MIN_MATCH_COUNT = 4    # to compute homography or for robust matching
best_match_name = None
best_good_matches = []
max_good = 0

for tpl in templates:
    if tpl["des"] is None:
        print(f"No descriptors in {tpl['name']}")
        continue
    matches = bf.knnMatch(tpl["des"], des_test, k=2)
    # Ratio test as per Lowe's paper
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    print(f"{tpl['name']}: {len(good)} good matches")
    if len(good) > max_good:
        max_good = len(good)
        best_match_name = tpl["name"]
        best_good_matches = good
        best_tpl = tpl

if best_match_name:
    print(f"\nBest match: {best_match_name} with {max_good} good matches")
    # Optionally, visualize matches
    img_matches = cv2.drawMatches(best_tpl["image"], best_tpl["kp"], test_img, kp_test, best_good_matches, None, flags=2)
    cv2.imshow('Best SIFT Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No good match found among templates.")