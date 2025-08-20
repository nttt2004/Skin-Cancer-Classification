$(function () {
  const $content = $("#main-content");
  const $spinner = $("#spinner");

  function setActive(link) {
    $(".nav-link").removeClass("active");
    $(link).addClass("active");
  }

  function loadContent(url, linkEl) {
    $spinner.removeClass("hidden");
    $.get(url)
      .done(function (html) {
        $content.html(html);
        if (linkEl) setActive(linkEl);
        window.scrollTo({ top: 0 });
      })
      .fail(function () {
        $content.html('<div class="alert">Không thể tải nội dung.</div>');
      })
      .always(function () {
        $spinner.addClass("hidden");
      });
  }

  // initial load
  loadContent("/dashboard_content", $(".nav-link[data-url='/dashboard_content']"));

  // nav click
  $(".nav-link").on("click", function (e) {
    e.preventDefault();
    const url = $(this).data("url");
    loadContent(url, this);
  });

  // delegate predict form submit (AJAX)
  $(document).on("submit", "#predict-form", function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    $spinner.removeClass("hidden");
    $.ajax({
      url: "/predict_content",
      type: "POST",
      data: formData,
      processData: false,
      contentType: false,
    })
      .done(function (html) {
        $content.html(html);
      })
      .fail(function () {
        alert("Tải/predict thất bại.");
      })
      .always(function () {
        $spinner.addClass("hidden");
      });
  });
});
