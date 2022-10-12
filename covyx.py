import cv2 as cv
import numpy as np

from Models.Results import Result


def is_starting_frame(frame, threshold: float) -> bool:
    """
    Determines if a given frame may be considered as
    the start of the pulmonar region within the current study
    """
    dimensions = frame.shape[0] * frame.shape[1]
    image = cv.threshold(frame, 110, 255, cv.THRESH_BINARY)[1]
    nonzero = cv.countNonZero(image)
    ratio = round((nonzero / dimensions), 4)
    return ratio >= threshold


def createMaskForFrame(frame):
    mask = cv.threshold(frame, 150, 255, cv.THRESH_BINARY)[1]
    cont, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i in cont:
        approx = cv.approxPolyDP(i, 0.000001 * cv.arcLength(i, True), True)
        cv.drawContours(mask, [approx], -1, 255, cv.FILLED)

    mask = cv.threshold(frame, 150, 255, cv.THRESH_BINARY)[1]
    return mask


def mask_frame(frame):
    """
    Uses the 'bitwise_or' operation in order to generate a mask
    for any given frame
    """
    mask = createMaskForFrame(frame)
    frame = cv.threshold(frame, 150, 255, cv.THRESH_BINARY)[1]
    frameMask = cv.bitwise_and(frame, mask)
    contour_list, _ = cv.findContours(
        frameMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    for contour in contour_list:
        approx = cv.approxPolyDP(
            contour, 0.0001 * cv.arcLength(contour, True), True
        )
        cv.drawContours(frameMask, [approx], -1, 255, cv.FILLED)

    return frameMask


def create_slate(width, height, color=True):
    """
    Creates a blank image with the provided dimensions
    """

    slate = np.zeros((width, height), dtype=np.uint8)
    if color:
        return cv.cvtColor(slate, cv.COLOR_GRAY2BGR)
    return slate


def resize_frame(frame, scale):
    """
    Returns an image scaled to a scale%
    of the original, conserving its aspect ratio
    """
    scale /= 100
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dim = (width, height)

    resize = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    return resize


def crop_frame(frame, area):
    """
    Returns a cropped portion given by an area => [x, y, w, h]
    """
    try:
        cropped = frame[
            area[1]: area[1] + area[3], area[0]: area[0] + area[2]
        ]
        return cropped
    except TypeError:
        return None


def get_frames(video_path: str, clean: bool = False) -> list:
    """
    Returns a list of all the frames in a video given its path,
    the 'clean' parameter determines if the frame should be de-noised
    before its analysis
    """
    video_file = cv.VideoCapture(video_path)

    fps = video_file.get(cv.CAP_PROP_FPS)

    frames = []
    has_frames, frame = video_file.read()

    dim = frame.shape[0] * frame.shape[1]

    # Scale if too big
    scale = 100
    if dim > 600000:
        scale = 50
    frame = resize_frame(frame, scale)

    while has_frames:
        frame = resize_frame(frame, scale)

        # Add small frame in order to
        # preserve info and facilitate border detection
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.copyMakeBorder(
            gray_frame, 5, 5, 5, 5, cv.BORDER_CONSTANT, value=(0)
        )

        if clean:
            gray_frame = cv.GaussianBlur(
                gray_frame, (3, 3), cv.BORDER_CONSTANT
            )

        frames.append(gray_frame)
        has_frames, frame = video_file.read()

    video_file.release()
    return frames, fps


def chop_image(image) -> tuple:
    """
    Returns a tuple containing both halves of an image cropped by the x axis
    """
    height, width = image.shape[:2]
    half_point_x = int(width * 0.5)

    images = []
    images.append(image[0:height, 0:half_point_x])
    images.append(image[0:height, half_point_x:width])
    return tuple(images)


# This list contains the frames that may cause trouble during analysis
_SKIPPED_FRAMES = []


def get_blob(
    img,
    median_bias=6,
    bilateral_bias=0,
    threshold=240,
    dilated=True,
    otsu=False,
):
    """
    Returns a binary map of the original image
    using Bilateral and Median blur to reduce noise
    """
    # Increase contrast
    alpha = np.array([2.2])
    beta = np.array([-50.0])
    contrast = img
    contrast = cv.add(contrast, beta)
    contrast = cv.multiply(contrast, alpha)

    BILATERAL_DIAMETER = 15 + bilateral_bias
    MEDIAN_KERNEL = 15 + median_bias

    blurred = cv.bilateralFilter(contrast, BILATERAL_DIAMETER, 80, 80)
    blurred = cv.medianBlur(blurred, MEDIAN_KERNEL)

    if otsu:
        blurred = cv.threshold(
            blurred, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )[1]
    else:
        blurred = cv.threshold(blurred, threshold, 255, cv.THRESH_BINARY)[1]

    canny = cv.Canny(blurred, 125, 175)
    if dilated:
        canny = cv.dilate(canny, (3, 5), iterations=3)

    return canny


def enum_children(hierarchy, index):
    """
    Returns a list with the indexes of the children
    for a value in a contour hierarchy
    """
    children = []
    c = hierarchy[1][0][index][2]  # Gets the index of the first child

    while c != -1:
        children.append(c)
        c = hierarchy[1][0][c][0]

    return tuple(children)


def get_contours(blob):
    """
    Returns an array with contours and hierarchies determined
    by cv.findContours with a TREE structure
    """
    # cont -> [contours, hierarchies]
    # hierarchy -> [next, previous, child, parent]
    cont = cv.findContours(blob, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    dont_draw = []
    for i in range(len(cont[0])):
        contour = cont[0][i]
        if len(contour) < 50:
            dont_draw.append(i)

    # print(f'contours found:  {len(cont[0])}')
    # print('dont draw: ', dont_draw)
    return cont


def get_dmg_ratio(img, mask) -> float:
    masked = cv.bitwise_and(mask, img)  # Total area
    limited = cv.threshold(masked, 180, 255, cv.THRESH_BINARY)[1]  # Damage

    total = cv.countNonZero(masked)
    afected = cv.countNonZero(limited)
    ratio = round(afected * 100 / total, 5)  # Damage for single frame
    return ratio


def get_equivalent_frame(target, framerate, starting_frame):
    return round(target * framerate) + starting_frame


def get_area_keyframes(frames, framerate, starting_frame=0):
    """
    Returns the keyframe value for
    breakpoints in 13 and 16 seconds
    """
    duration = len(frames) / framerate
    offset = starting_frame / framerate
    keys = [0]
    # 13 y 16 s para 6 secciones y pulmón derecho en 5 secciones
    # 16 s para pulmón izquierdo en 5 secciones
    if (duration - offset) < 18:
        key_6 = round((len(frames) / 3) + starting_frame)
        key_5 = round((len(frames) / 2) + starting_frame)
    else:
        key_6 = round(13 * framerate) + starting_frame
        key_5 = round(16 * framerate) + starting_frame

    keys.append(key_6)
    keys.append(key_5)
    return tuple(keys)


def draw_contours(frame, contours):
    """
    Draws the contours from a get_contours return over the provided image
    """
    try:
        contoured_frame = frame
        for i in range(len(contours[1][0])):
            h = contours[1][0][i]
            children = enum_children(contours, i)

            # if no valid children
            if (
                (len(children) < 2 and i > 2) or (h[2] in _SKIPPED_FRAMES)
            ) and (i not in _SKIPPED_FRAMES):
                cv.drawContours(
                    contoured_frame, contours[0], i, (255), cv.FILLED
                )

        return contoured_frame
    except TypeError as e:
        print(
            "Type Error in draw_contours, received:", type(frame), "\n\t\t", e
        )
        return frame


def get_damage_ratio(anomaly_map, area_map, original_area_map=None):
    """
    Returns the damage ratio for an frame.
    """
    total = cv.countNonZero(area_map)  # Lung area
    afected = cv.countNonZero(anomaly_map)  # Anomalies

    try:
        ratio = round(afected * 100 / total, 5)
        if original_area_map is not None:
            original_total = cv.countNonZero(original_area_map)

            ratio = ratio * (total / original_total)

    except ZeroDivisionError:
        ratio = 0

    return ratio


def get_start(frames, threshold=0.1, inverted=False):
    """
    Returns the initial analysis index if found,
    else returns 0.
    """
    for i, frame in enumerate(frames):
        mask = mask_frame(frame)
        if inverted:
            masked = cv.bitwise_and(mask, frame)
        else:
            masked = cv.bitwise_and(mask, (255 - frame))
        if is_starting_frame(masked, threshold):
            return i
    return 0


def get_image_maps(frame):
    """
    Returns a tuple with the found area mask,
    the masked original frame and its corresponding
    damage map.
    """
    blob = get_blob(frame)
    cont = get_contours(blob)

    slate = create_slate(frame.shape[0], frame.shape[1], color=False)
    mask = draw_contours(slate, cont)

    mask = cv.GaussianBlur(mask, (15, 15), cv.BORDER_DEFAULT)
    mask = cv.threshold(mask, 254, 255, cv.THRESH_BINARY)[1]

    masked = cv.bitwise_and(mask, frame)

    limited = cv.threshold(masked, 68, 255, cv.THRESH_BINARY)[1]
    return tuple([mask, masked, limited])


def get_time_data(frames, framerate):
    """
    Returns a tuple containing the starting
    and ending frame in a frame collection
    """
    start_frame = get_start(frames, 0.045)
    end_frame = (36 * framerate) + start_frame
    if end_frame > len(frames):
        end_frame = len(frames) - 1
    return tuple([start_frame, end_frame])


def get_total_analyzed_frames(sections):
    frames = []
    for section in sections:
        frames.append(sections[section][1])

    return tuple([sum(frames), frames])


def analyze_video(frames, framerate=10) -> Result:
    start_frame, end_frame = get_time_data(frames, framerate)
    keyframes = get_area_keyframes(
        frames, framerate, starting_frame=start_frame
    )

    # Represents the 3 sections, contains:
    #   total damage, frames used,
    #   left damage over total area,
    #   left damage & right damage
    sections = {0: [0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0], 2: [0, 0, 0, 0, 0]}

    section_index = -1
    for i, f in enumerate(frames):  # Analyzes all available frames
        if i in keyframes:
            section_index += 1
        if i < start_frame:
            continue
        elif i > end_frame:
            break

        mask, masked, limited = get_image_maps(f)

        current_total_dmg = get_damage_ratio(limited, mask)
        current_total_dmg = round(current_total_dmg, 5)

        if current_total_dmg < 3:
            current_total_dmg = 0.01
            current_left_dmg = 0.005
            local_left_dmg = 0.01
            local_right_dmg = 0.01
        else:
            chopped_lung = chop_image(mask)  # Tuples with half of an image
            chopped_anomalies = chop_image(limited)

            current_left_dmg = get_damage_ratio(
                chopped_anomalies[0], chopped_lung[0], original_area_map=mask
            )

            current_left_dmg = round(current_left_dmg, 5)

            local_left_dmg = get_damage_ratio(
                chopped_anomalies[0], chopped_lung[0]
            )
            local_right_dmg = get_damage_ratio(
                chopped_anomalies[1], chopped_lung[1]
            )

        sections[section_index][0] += current_total_dmg  # Total damage
        sections[section_index][1] += 1  # Analyzed frames
        sections[section_index][2] += (
            current_left_dmg / current_total_dmg
        )  # Left distribution
        sections[section_index][3] += round(local_left_dmg, 3)  # Left damage
        sections[section_index][4] += round(local_right_dmg, 3)  # Right damage

    total_frames, sections_frames = get_total_analyzed_frames(sections)
    result = 0
    sections_results = []
    sections_left_and_right = []
    section_ratios = []

    for section in sections:  # Evaluates damage found
        left_and_right = []
        damage_and_percentage = []

        total_section_ratio = sections[section][0]
        total_section_frames = sections[section][1]

        try:
            res = total_section_ratio / total_section_frames
            damage_and_percentage.append(round(res, 5))  # Local section damage

        except ZeroDivisionError:
            damage_and_percentage.append(0)  # Local section damage

        percent = total_section_frames / total_frames
        damage_and_percentage.append(
            percent
        )  # Percentage of section for whole study
        sections_results.append(damage_and_percentage)

        try:
            section_left = sections[section][2]
            section_left /= total_section_frames
            left_percentage = int(section_left * 100)
            right_percentage = 100 - left_percentage

            left_percentage /= 100
            right_percentage /= 100

            left_and_right.append(left_percentage)
            left_and_right.append(right_percentage)

            sections_left_and_right.append(left_and_right)
        except ZeroDivisionError:
            left_and_right.append(0)
            left_and_right.append(0)

            sections_left_and_right.append(left_and_right)

        try:
            left_dmg = sections[section][3]
            right_dmg = sections[section][4]

            left_dmg /= total_section_frames
            right_dmg /= total_section_frames

            section_ratios.append(round(left_dmg, 5))
            section_ratios.append(round(right_dmg, 5))
        except ZeroDivisionError:
            section_ratios.append(0)
            section_ratios.append(0)

    for i in range(len(sections_results)):
        result += sections_results[i][0] * sections_results[i][1]
        sections_results[i][1] = round(sections_results[i][1], 2)

    analysis_results = Result(
        result,
        sections_results,
        sections_left_and_right,
        sections_frames,
        section_ratios,
    )

    return analysis_results


def analyze(video_path: str):
    frames, framerate = get_frames(video_path)
    return analyze_video(frames, framerate)
