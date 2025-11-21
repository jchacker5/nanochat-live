#!/bin/bash
# Docker helper script for SRGI/NanoChat

set -e

CONTAINER_NAME="nanochat-srgi"
IMAGE_NAME="nanochat-srgi"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_info() {
    echo -e "${YELLOW}ℹ $1${NC}"
}

function build_image() {
    print_header "Building Docker Image"
    # Try docker compose (newer) first, then docker-compose (older)
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        docker compose build
    elif command -v docker-compose &> /dev/null; then
        docker-compose build
    else
        echo "Error: Neither 'docker compose' nor 'docker-compose' found"
        exit 1
    fi
    print_success "Image built successfully"
}

# Helper function to get docker compose command
function get_docker_compose() {
    if command -v docker &> /dev/null && docker compose version &> /dev/null; then
        echo "docker compose"
    elif command -v docker-compose &> /dev/null; then
        echo "docker-compose"
    else
        echo "ERROR"
    fi
}

function start_container() {
    print_header "Starting Container"
    DOCKER_COMPOSE=$(get_docker_compose)
    if [ "$DOCKER_COMPOSE" = "ERROR" ]; then
        echo "Error: Docker Compose not found"
        exit 1
    fi
    $DOCKER_COMPOSE up -d srgi
    print_success "Container started"
    print_info "Run 'docker-helper.sh shell' to enter the container"
}

function stop_container() {
    print_header "Stopping Container"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE down
    print_success "Container stopped"
}

function shell() {
    print_header "Entering Container Shell"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE exec srgi /bin/bash
}

function run_tests() {
    print_header "Running Tests"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE exec srgi pytest tests/ -v
}

function run_ebm_tests() {
    print_header "Running EBM Tests"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE exec srgi pytest tests/test_ebm_hopfield.py -v
}

function install_thrml() {
    print_header "Installing THRML"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE exec srgi pip install git+https://github.com/extropic-ai/thrml.git
    print_success "THRML installed"
}

function clean() {
    print_header "Cleaning Up"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE down -v
    docker rmi $IMAGE_NAME 2>/dev/null || true
    print_success "Cleaned up containers and images"
}

function logs() {
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE logs -f srgi
}

function rebuild() {
    print_header "Rebuilding Container"
    DOCKER_COMPOSE=$(get_docker_compose)
    $DOCKER_COMPOSE build --no-cache
    print_success "Container rebuilt"
}

function show_help() {
    echo "SRGI/NanoChat Docker Helper"
    echo ""
    echo "Usage: ./docker-helper.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build          Build the Docker image"
    echo "  start          Start the container"
    echo "  stop           Stop the container"
    echo "  shell          Enter the container shell"
    echo "  test           Run all tests"
    echo "  ebm-test       Run EBM-specific tests"
    echo "  install-thrml  Install THRML in container"
    echo "  logs           Show container logs"
    echo "  rebuild        Rebuild container from scratch"
    echo "  clean          Remove containers and images"
    echo "  help           Show this help message"
}

# Main command handler
case "$1" in
    build)
        build_image
        ;;
    start)
        start_container
        ;;
    stop)
        stop_container
        ;;
    shell)
        shell
        ;;
    test)
        run_tests
        ;;
    ebm-test)
        run_ebm_tests
        ;;
    install-thrml)
        install_thrml
        ;;
    logs)
        logs
        ;;
    rebuild)
        rebuild
        ;;
    clean)
        clean
        ;;
    help|--help|-h|"")
        show_help
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

