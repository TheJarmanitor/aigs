# %% utils.py

# %% Visualization
def animate(board_seq, filename):
    def board_to_svg(board):
        rows = len(board)
        cols = len(board[0])
        cell_size = 10  # Size of each cell in the SVG
        svg_elements = []

        for r in range(rows):
            for c in range(cols):
                if board[r][c] == 1:
                    x = c * cell_size
                    y = r * cell_size
                    svg_elements.append(
                        f'<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" fill="black" />'
                    )

        return "\n".join(svg_elements)

    def generate_svg_animation(board_seq):
        rows = len(board_seq[0])
        cols = len(board_seq[0][0])
        cell_size = 10
        width = cols * cell_size
        height = rows * cell_size
        duration = 0.5  # Duration of each frame in seconds

        svg_header = (
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" '
            f'xmlns:xlink="http://www.w3.org/1999/xlink">\n'
        )
        svg_footer = "</svg>"
        svg_content = []

        for i, board in enumerate(board_seq):
            svg_content.append(f'<g id="frame{i}" style="display:none">\n{board_to_svg(board)}\n</g>')

        svg_content.append(
            '<animate xlink:href="#frame0" attributeName="display" values="inline;none" dur="0.5s" repeatCount="indefinite" />'
        )

        for i in range(1, len(board_seq)):
            svg_content.append(
                f'<animate xlink:href="#frame{i}" attributeName="display" values="none;inline" begin="{i * duration}s" dur="1s" repeatCount="indefinite" />'
            )

        return svg_header + "\n".join(svg_content) + "\n" + svg_footer

    svg = generate_svg_animation(board_seq)

    with open(filename, "w") as f:
        f.write(svg)
