function SwitchButton(id, keys, default_key_index) {
    this.id = id;
    this.keys = keys;
    this.state = default_key_index;
    $(`#${id}`).addClass("SwitchButton");
    for(let i = 0; i < keys.length; i++) {
        let is_default = (i === default_key_index) ? "Active" : "";
        $(`#${id}`).append(`<div class="SwitchButtonKey ${id} ${is_default}" id="SwitchButtonKey_${id}_${i}">${this.keys[i]}</div>`);
    }
    let _this = this;
    for(let i = 0; i < keys.length; i++) {
        $(`#SwitchButtonKey_${id}_${i}`).on("click", function(e) {
            $(`.SwitchButtonKey.${id}`).removeClass("Active");
            $(this).addClass("Active");
            _this.state = i;
            e.stopPropagation();
            return false;
        });
    }
}

SwitchButton.prototype.getKey = function() {
    return this.keys[this.state];
};

function RangeInput(id, title, min, max, step, value) {
    this.id = id;
    $(`#${id}`).addClass("RangeInput");
    $(`#${id}`).append(`<div class="RangeInputTitle" id="RangeInputTitle_${id}">${title}</div>`);
    $(`#${id}`).append(`<input type="range" class="RangeInputControl" id="RangeInputControl_${id}" min="${min}" max="${max}" step="${step}" value="${value}" />`);
    $(`#${id}`).append(`<div class="RangeInputValue" id="RangeInputValue_${id}">${value}</div>`);
    $(`#RangeInputControl_${id}`).on("input", function(e) {
        $(`#RangeInputValue_${id}`).html($(this).val());
        e.stopPropagation();
        return false;
    });
}

RangeInput.prototype.getValue = function() {
    return $(`#RangeInputControl_${this.id}`).val();
}

function Modal(id, maskId) {
    this.id = id;
    this.maskId = maskId;
    this.node = $(`#${id}`);
    this.maskNode = $(`#${maskId}`);
    this.state = this.node.css("display");
}
Modal.prototype.show = function() {
    this.node.css("display", "flex");
    this.maskNode.show();
    this.state = "flex";
    $(".Main").css("filter", "blur(10px)");
}
Modal.prototype.hide = function() {
    this.node.css("display", "none");
    this.maskNode.hide();
    this.state = "none";
    $(".Main").css("filter", "none");
}
Modal.prototype.isHidden = function() {
    return (this.state === "none") ? true : false;
}
Modal.prototype.toggle = function() {
    if(this.isHidden() === true) {
        this.show();
    }
    else {
        this.hide();
    }
}